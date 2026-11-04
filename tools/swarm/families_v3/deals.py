"""Deal packs for extract3: a deal desk turns a contract pack into one strict deal record.

Each pack is a master agreement or framework plus signed order documents, amendments
and side letters, together with documents that must not change the record: unsigned
drafts, "subject to contract" emails, an affiliate's NDA, a rate card, another
customer's order. The brief states the deal desk's definitions (order of precedence,
end date, notice deadline, contract value), as a real one would; every field of the
record is one requirement.
"""

from __future__ import annotations

import calendar
import datetime as dt
from decimal import ROUND_HALF_UP, Decimal

from families import synth, tool

US = "Northwind Analytics Ltd"
LAWS = ["England and Wales", "Scotland", "Ireland", "the Netherlands", "Germany"]
SUFFIX = ["Ltd", "Limited", "GmbH", "B.V.", "S.A."]
PRODUCTS = ["Analytics Platform", "Data Connectors", "Premium Support", "Audit Log Archive", "Sandbox Environments",
            "Single Sign-On", "Forecasting Module", "API Access"]
PARTS = ["polymer housing", "M8 hex bolt", "gasket ring 40 mm", "aluminium bracket", "control board v3",
         "sensor cable 2 m", "valve seat", "filter cartridge"]
INCOTERMS = ["DAP Rotterdam", "DAP Hamburg", "FCA Antwerp", "DDP Manchester", "CPT Felixstowe"]
ROLES = ["Engagement Lead", "Solution Architect", "Data Engineer", "QA Lead", "Delivery Manager"]


def d2(x) -> Decimal:
    return Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def add_months(day: dt.date, months: int) -> dt.date:
    y, m = divmod(day.month - 1 + months, 12)
    year, month = day.year + y, m + 1
    return day.replace(year=year, month=month, day=min(day.day, calendar.monthrange(year, month)[1]))


def long_date(day: dt.date) -> str:
    return f"{day.day} {day.strftime('%B %Y')}"


def boiler(rng, n: int) -> str:
    return synth.boilerplate(rng, n)


def signature(rng, parties: list[tuple[str, str, dt.date]]) -> str:
    """Signature block: (company, signatory, date) per party."""
    rows = []
    for company, person, date in parties:
        rows.append(f"Signed for and on behalf of {company}\nName: {person}\nTitle: "
                    f"{rng.choice(['Director', 'Chief Financial Officer', 'Head of Procurement', 'Managing Director'])}"
                    f"\nDate: {date.isoformat()}\nSignature: /s/ {person}")
    return "\n\n".join(rows)


def unsigned(company: str) -> str:
    return (f"DRAFT - NOT FOR SIGNATURE. This draft is circulated for discussion only and has not been executed by "
            f"{US} or {company}.")


class Pack:
    """Documents (kind, title, date, text), the expected record and each field's type."""

    def __init__(self):
        self.docs: list[tuple[str, str, dt.date, str]] = []
        self.expected: dict = {}
        self.types: dict[str, str] = {}
        self.brief = ""

    def doc(self, kind: str, title: str, date: dt.date, text: str) -> None:
        self.docs.append((kind, title, date, text))

    def field(self, name: str, kind: str, value) -> None:
        self.types[name] = kind
        self.expected[name] = value


def parties(rng, taken: set) -> tuple[str, str, str, str]:
    base = synth.company(rng, taken)
    supplier = f"{base} {rng.choice(SUFFIX)}"
    affiliate = f"{base} Holdings {rng.choice(['Ltd', 'Limited', 'Inc.'])}"
    return base, supplier, affiliate, f"{rng.randrange(10**7, 10**8)}"


def other_contracts(rng, pack: Pack, taken: set, day: dt.date) -> None:
    """Agreements with other suppliers in the same archive."""
    for _ in range(rng.randrange(2, 4)):
        other = f"{synth.company(rng, taken)} {rng.choice(SUFFIX)}"
        when = day - dt.timedelta(days=rng.randrange(-200, 400))
        pack.doc("agreement", f"{rng.choice(['Services', 'Supply', 'Master Subscription'])} Agreement - {other} and {US}",
                 when, f"AGREEMENT between {other} and {US}\n\n1. Term. {rng.choice([12, 24, 36])} months from "
                 f"{long_date(when)}.\n2. Payment. Within {rng.choice([30, 45, 60])} days.\n3. Liability. Capped at "
                 f"{rng.choice(['GBP', 'EUR', 'USD'])} {rng.randrange(50, 900) * 1000:,}.\n4. Governing law. The laws of "
                 f"{rng.choice(LAWS)}.\n\n{boiler(rng, 500)}\n\n"
                 + signature(rng, [(other, synth.person(rng, set()), when), (US, synth.person(rng, set()), when)]))


def affiliate_nda(rng, pack: Pack, affiliate: str, day: dt.date) -> None:
    number = f"{rng.randrange(10**7, 10**8)}"
    pack.doc("nda", f"Mutual NDA - {affiliate}", day,
             f"MUTUAL NON-DISCLOSURE AGREEMENT\n\nBetween {US} and {affiliate} (company number {number}).\n\n"
             f"1. Each party will keep the other's confidential information confidential for three years.\n"
             f"2. This agreement is governed by the laws of {rng.choice(LAWS)}.\n\n{boiler(rng, 150)}\n\n"
             + signature(rng, [(US, synth.person(rng, set()), day), (affiliate, synth.person(rng, set()), day)]))


# --------------------------------------------------------------------- SaaS --

def table(desc: str, key: str, **columns: str) -> tuple:
    """A table field: rows of named columns, matched by their `key` column (types: string, date, integer, number)."""
    return ("table", desc, key, columns)


SAAS_FIELDS = {
    "supplier_legal_name": ("string", "the supplier's full legal name"),
    "supplier_company_number": ("string", "the supplier's registered company number"),
    "effective_date": ("string", "YYYY-MM-DD"),
    "initial_term_months": ("integer", "as amended"),
    "end_date": ("string", "YYYY-MM-DD, last day of the initial term"),
    "renewal_term_months": ("integer", "length of each automatic renewal"),
    "notice_period_days": ("integer", "non-renewal notice period"),
    "notice_deadline": ("string", "YYYY-MM-DD"),
    "governing_law": ("string", "the governing law, e.g. England and Wales"),
    "currency": ("string", "ISO code"),
    "billing_frequency": ("string", "annual, quarterly or monthly"),
    "payment_terms_days": ("integer", "days from invoice date"),
    "lines": table("the subscription lines in force on the reference date, before any uplift", "product",
                   product="string", units="integer", unit_price="number", discount_pct="number", annual_fee="number"),
    "uplift_pct": ("number", "annual fee uplift in percent"),
    "first_uplift_date": ("string", "YYYY-MM-DD, the first anniversary of the effective date"),
    "annual_fee_year1": ("number", "annual fee in the first contract year"),
    "annual_fee_current": ("number", "annual fee in force on the reference date"),
    "total_contract_value": ("number", "fees over the initial term as amended"),
    "liability_cap": ("number", "the liability cap amount"),
    "customer_signatory": ("string", f"who signed the order form for {US}"),
}


def saas(rng, taken: set) -> Pack:
    pack = Pack()
    base, supplier, affiliate, number = parties(rng, taken)
    cur = rng.choice(["GBP", "EUR", "USD"])
    law = rng.choice(LAWS)
    signed_supplier = dt.date(rng.choice([2024, 2025]), rng.randrange(1, 13), rng.randrange(1, 29))
    signed_us = signed_supplier + dt.timedelta(days=rng.randrange(0, 12))
    effective = max(signed_supplier, signed_us)
    term = rng.choice([24, 36])
    uplift = Decimal(rng.choice(["3", "4", "5", "6"]))
    renew = rng.choice([12, 24])
    notice = rng.choice([30, 60, 90])
    msa_pay = rng.choice([30, 45])
    of_pay = rng.choice([None, 45, 60]) if msa_pay == 30 else rng.choice([None, 30, 60])
    cap = rng.randrange(50, 500) * 1000
    freq = rng.choice(["annual", "quarterly", "monthly"])
    us_person, them_person = synth.person(rng, taken), synth.person(rng, taken)
    lines = []
    for product in rng.sample(PRODUCTS, rng.randrange(3, 5)):
        lines.append({"product": product, "qty": rng.randrange(5, 200), "price": d2(rng.randrange(800, 12000) / 100),
                      "disc": rng.choice([0, 5, 10, 15, 20])})

    def fee(rows) -> Decimal:
        return sum((d2(Decimal(r["qty"]) * r["price"] * (100 - r["disc"]) / 100) for r in rows), Decimal(0))

    pack.doc("agreement", f"Master Subscription Agreement - {supplier} and {US}", effective,
             f"MASTER SUBSCRIPTION AGREEMENT\n\nThis Agreement is made between {supplier}, a company registered under "
             f"number {number} (the \"Supplier\"), and {US} (the \"Customer\").\n\n"
             f"1. Commencement. This Agreement takes effect on the date of the last signature below (the "
             f"\"Effective Date\").\n"
             f"2. Term. The Subscription starts on the Effective Date and continues for the Initial Term stated in "
             f"the Order Form. It then renews automatically for successive periods of {renew} months unless either "
             f"party gives written notice of non-renewal at least {notice} days before the end of the then-current "
             f"term.\n"
             f"3. Fees. The Customer pays the fees in the Order Form in {cur}.\n"
             f"4. Payment. Invoices are payable within {msa_pay} days of the invoice date.\n"
             f"5. Liability. Each party's total aggregate liability under this Agreement is limited to {cur} "
             f"{cap:,}.\n"
             f"5A. Uplift. On each anniversary of the Effective Date the annual fees increase by {uplift}%.\n"
             f"6. Governing law. This Agreement is governed by the laws of {law}.\n"
             f"7. Precedence. If documents conflict, a signed Amendment prevails over the Order Form, and the Order "
             f"Form prevails over this Agreement. Side letters signed by both parties form part of this Agreement "
             f"and prevail over the Order Form.\n\n{boiler(rng, 1800)}\n\n"
             + signature(rng, [(supplier, them_person, signed_supplier), (US, us_person, signed_us)]))
    table = "\n".join(f" {r['product']:<22} | {r['qty']:>5} | {cur} {r['price']:>9} | {r['disc']:>3}%" for r in lines)
    of_signatory = synth.person(rng, taken)
    pack.doc("order form", f"Order Form OF-1 - {supplier}", signed_us,
             f"ORDER FORM OF-1 under the Master Subscription Agreement between {supplier} and {US}\n\n"
             f"Initial Term: {term} months\nBilling: {freq}, in advance\nCurrency: {cur}\n"
             + (f"Payment terms: {of_pay} days from the invoice date (this replaces the payment clause of the "
                f"Agreement)\n" if of_pay else "")
             + f"\n Product                | Units | Unit price/yr | Discount\n{table}\n\n"
             f"Fees are units x unit price per year, less the discount.\n\n{boiler(rng, 400)}\n\n"
             + signature(rng, [(supplier, them_person, signed_supplier), (US, of_signatory, signed_us)]))
    # amendment 1: more units from a later month, maybe a longer term or a new notice period
    k = rng.randrange(3, term - 1)
    amend_from = add_months(effective, k)
    bumped = rng.choice(lines)
    extra = rng.randrange(5, 60)
    new_term = term + 12 if rng.random() < 0.5 else term
    new_notice = rng.choice([n for n in (30, 60, 90) if n != notice]) if rng.random() < 0.4 else notice
    after = [dict(r, qty=r["qty"] + extra) if r is bumped else dict(r) for r in lines]
    changes = [f"(a) With effect from {amend_from.isoformat()}, the units of {bumped['product']} increase by "
               f"{extra} to {bumped['qty'] + extra}, at the same unit price and discount."]
    if new_term != term:
        changes.append(f"(b) The Initial Term is extended to {new_term} months.")
    if new_notice != notice:
        changes.append(f"({'c' if new_term != term else 'b'}) The non-renewal notice period in clause 2 of the "
                       f"Agreement becomes {new_notice} days.")
    signed_amend = amend_from - dt.timedelta(days=rng.randrange(5, 30))
    pack.doc("amendment", f"Amendment No. 1 - {supplier}", signed_amend,
             f"AMENDMENT NO. 1 to the Master Subscription Agreement and Order Form OF-1 between {supplier} and {US}\n\n"
             + "\n".join(changes) + f"\nAll other terms are unchanged.\n\n{boiler(rng, 300)}\n\n"
             + signature(rng, [(supplier, them_person, signed_amend), (US, us_person, signed_amend)]))
    # a signed side letter with an extra discount on one product, from the start
    if rng.random() < 0.6:
        favoured = rng.choice(lines)
        bonus = rng.choice([5, 10])
        for rows in (lines, after):
            for r in rows:
                if r["product"] == favoured["product"]:
                    r["disc"] += bonus
        side_day = signed_us + dt.timedelta(days=rng.randrange(0, 5))
        pack.doc("side letter", f"Side letter - {favoured['product']} discount", side_day,
                 f"SIDE LETTER to the Master Subscription Agreement between {supplier} and {US}\n\nThe parties agree "
                 f"that the discount on {favoured['product']} is increased by {bonus} percentage points for the "
                 f"whole Initial Term.\n\n" + signature(rng, [(supplier, them_person, side_day), (US, us_person, side_day)]))
    # documents that change nothing
    draft_day = amend_from + dt.timedelta(days=rng.randrange(20, 60))
    pack.doc("amendment", f"Amendment No. 2 - {supplier} (draft)", draft_day,
             f"AMENDMENT NO. 2 to the Master Subscription Agreement between {supplier} and {US}\n\n{unsigned(supplier)}"
             f"\n\n(a) The unit price of every product decreases by 10%.\n(b) The Initial Term is extended by a "
             f"further 12 months.\n(c) Payment terms become 90 days.\n\n{boiler(rng, 250)}")
    pack.doc("email", f"RE: pricing for {rng.choice(lines)['product']}", signed_us - dt.timedelta(days=rng.randrange(5, 20)),
             f"From: {them_person} <sales@{base.split()[0].lower()}.example>\nTo: {us_person}\n\nHappy to offer a "
             f"further 15% off every line and a liability cap of {cur} {cap * 2:,} if you sign this month. Subject to "
             f"contract.\n\nBest regards,\n{them_person}")
    pack.doc("rate card", f"{supplier} list prices {effective.year}", effective - dt.timedelta(days=90),
             f"{supplier.upper()} - PUBLIC LIST PRICES {effective.year}\n\n" + "\n".join(
                 f" {p:<22} {cur} {d2(rng.randrange(900, 15000) / 100)} per unit per year" for p in PRODUCTS)
             + "\n\nList prices are subject to change and do not apply to signed order forms.")
    affiliate_nda(rng, pack, affiliate, effective - dt.timedelta(days=rng.randrange(30, 120)))
    other_contracts(rng, pack, taken, effective)
    end = add_months(effective, new_term) - dt.timedelta(days=1)
    before, after_fee = fee(lines), fee(after)
    pack.field("supplier_legal_name", "name", supplier)
    pack.field("supplier_company_number", "code", number)
    pack.field("effective_date", "date", effective.isoformat())
    pack.field("initial_term_months", "int", new_term)
    pack.field("end_date", "date", end.isoformat())
    pack.field("renewal_term_months", "int", renew)
    pack.field("notice_period_days", "int", new_notice)
    pack.field("notice_deadline", "date", (end - dt.timedelta(days=new_notice)).isoformat())
    pack.field("governing_law", "name", law)
    pack.field("currency", "code", cur)
    pack.field("billing_frequency", "name", freq)
    pack.field("payment_terms_days", "int", of_pay or msa_pay)
    def uplifted(annual: Decimal, years: int) -> Decimal:
        for _ in range(years):
            annual = d2(annual * (100 + uplift) / 100)
        return annual
    today = add_months(effective, rng.randrange(13, new_term - 1)) + dt.timedelta(days=rng.randrange(0, 20))
    years_in = sum(1 for n in range(1, new_term // 12 + 1) if add_months(effective, 12 * n) <= today)
    current = uplifted(after_fee if today >= amend_from else before, years_in)
    tcv = sum((uplifted(after_fee if m >= k else before, m // 12) for m in range(new_term)), Decimal(0)) / 12
    in_force = after if today >= amend_from else lines
    pack.field("lines", "table", [{"product": r["product"], "units": r["qty"], "unit_price": float(r["price"]),
                                   "discount_pct": float(r["disc"]),
                                   "annual_fee": float(d2(Decimal(r["qty"]) * r["price"] * (100 - r["disc"]) / 100))}
                                  for r in in_force])
    pack.field("uplift_pct", "number", float(uplift))
    pack.field("first_uplift_date", "date", add_months(effective, 12).isoformat())
    pack.field("annual_fee_year1", "money", float(before))
    pack.field("annual_fee_current", "money", float(current))
    pack.field("total_contract_value", "money", float(d2(tcv)))
    pack.field("liability_cap", "money", float(cap))
    pack.field("customer_signatory", "name", of_signatory)
    pack.brief = (
        f"Create the deal record for our software subscription with {base}; the reference date is {today.isoformat()}. Deal desk rules: only documents signed "
        f"by both parties count (drafts, emails and list prices do not change the deal); a signed amendment prevails "
        f"over the order form, which prevails over the master agreement; side letters signed by both parties apply. "
        f"end_date is the effective date plus the initial term (as amended) minus one day; notice_deadline is "
        f"end_date minus the notice period in days. A line's annual fee is units x unit price less its discount, "
        f"rounded to 2 decimals; the annual fee is the sum over lines. The uplift raises the whole annual fee on "
        f"each anniversary, compounding, rounded to 2 decimals each time. annual_fee_year1 is the fee at the "
        f"effective date; annual_fee_current is the fee in force on the reference date with every signed change applied; "
        f"lines lists the lines in force on that date at list basis (no uplift), one row per product. "
        f"total_contract_value covers the initial term as amended: each month of the term costs one twelfth of the "
        f"annual fee in force that month (a change takes effect from its stated date), summed and rounded to 2 "
        f"decimals at the end. "
        f"Use the supplier's own legal name and number, not an affiliate's.")
    return pack


# ------------------------------------------------------------------- supply --

SUPPLY_FIELDS = {
    "supplier_legal_name": ("string", "the supplier's full legal name"),
    "supplier_vat_number": ("string", "the supplier's VAT number"),
    "commencement_date": ("string", "YYYY-MM-DD"),
    "term_months": ("integer", "as amended"),
    "expiry_date": ("string", "YYYY-MM-DD, last day of the term"),
    "incoterm": ("string", "delivery term with place, e.g. DAP Rotterdam"),
    "currency": ("string", "ISO code"),
    "payment_terms_days": ("integer", "days from invoice date"),
    "minimum_annual_volume": ("integer", "units per contract year"),
    "forecast_annual_volume": ("integer", "units, from the current signed forecast"),
    "unit_price_at_forecast": ("number", "unit price of the tier the forecast falls in"),
    "price_tiers": table("the price tiers as amended; to_units is 0 for the open-ended top tier", "from_units",
                         from_units="integer", to_units="integer", unit_price="number"),
    "estimated_annual_value": ("number", "forecast volume x that unit price"),
    "rebate_threshold_units": ("integer", "annual units above which the volume rebate applies"),
    "rebate_pct": ("number", "volume rebate in percent"),
    "estimated_rebate": ("number", "rebate on the estimated annual value, 0 if the forecast is not above the threshold"),
    "indexation_pct": ("number", "annual price adjustment in percent"),
    "year2_unit_price": ("number", "unit price at forecast after one indexation"),
    "late_penalty_pct_per_week": ("number", "late delivery penalty in percent per week"),
    "late_penalty_cap_pct": ("number", "cap on late delivery penalties in percent"),
    "warranty_months": ("integer", "warranty period"),
    "exclusive_supply": ("boolean", "whether we must buy these parts only from this supplier"),
    "governing_law": ("string", "the governing law"),
}


def supply(rng, taken: set) -> Pack:
    pack = Pack()
    base, supplier, affiliate, _ = parties(rng, taken)
    vat = f"{rng.choice(['GB', 'NL', 'DE', 'IE'])}{rng.randrange(10**8, 10**9)}"
    part = rng.choice(PARTS)
    cur = rng.choice(["GBP", "EUR", "USD"])
    law = rng.choice(LAWS)
    commence = dt.date(rng.choice([2024, 2025]), rng.randrange(1, 13), 1)
    term = rng.choice([24, 36, 48])
    inco = rng.choice(INCOTERMS)
    pay = rng.choice([30, 45, 60])
    moq = rng.randrange(5, 40) * 1000
    bounds = sorted(rng.sample(range(10, 120), 2))
    tiers = [(0, bounds[0] * 1000 - 1), (bounds[0] * 1000, bounds[1] * 1000 - 1), (bounds[1] * 1000, None)]
    top = rng.randrange(300, 2000)
    prices = [d2(top / 100), d2((top - rng.randrange(10, 40)) / 100)]
    prices.append(d2(prices[1] - Decimal(rng.randrange(5, 30)) / 100))
    index_pct = Decimal(rng.choice(["1.5", "2", "2.5", "3", "3.5"]))
    penalty, pen_cap = Decimal(rng.choice(["0.5", "1", "1.5", "2"])), Decimal(rng.choice(["5", "8", "10", "15"]))
    warranty = rng.choice([12, 18, 24, 36])
    exclusive = rng.random() < 0.5
    us_person, them_person = synth.person(rng, taken), synth.person(rng, taken)
    signed = commence - dt.timedelta(days=rng.randrange(10, 40))

    def tier_text(rows):
        return "\n".join(f" {lo:>7,} - {hi:>7,} units: {cur} {p} per unit" if hi is not None else
                         f" {lo:>7,} units and above: {cur} {p} per unit" for (lo, hi), p in zip(tiers, rows))
    pack.doc("agreement", f"Supply Agreement - {supplier} and {US}", signed,
             f"SUPPLY AGREEMENT\n\nBetween {supplier} (VAT {vat}) (the \"Supplier\") and {US} (the \"Buyer\") for the "
             f"supply of {part}s.\n\n"
             f"1. Term. This Agreement commences on {long_date(commence)} (the \"Commencement Date\") and continues "
             f"for {term} months.\n"
             f"2. Prices. Unit prices depend on the Buyer's annual volume; the price of the tier the whole annual "
             f"volume falls in applies to all units:\n{tier_text(prices)}\n"
             f"3. Indexation. On each anniversary of the Commencement Date all unit prices increase by "
             f"{index_pct}%, rounded to 2 decimals.\n"
             f"4. Minimum volume. The Buyer will order at least {moq:,} units per contract year.\n"
             f"5. Delivery. {inco} (Incoterms 2020).\n"
             f"6. Payment. Invoices are payable in {cur} within {pay} days of the invoice date.\n"
             f"7. Late delivery. For each full week of late delivery the Supplier pays {penalty}% of the value of the "
             f"late goods, capped at {pen_cap}%.\n"
             f"8. Warranty. The goods are warranted for {warranty} months from delivery.\n"
             + (f"9. Exclusivity. During the term the Buyer purchases {part}s exclusively from the Supplier.\n" if exclusive
                else f"9. Non-exclusivity. The Buyer may purchase {part}s from other suppliers.\n")
             + f"10. Governing law. The laws of {law}.\n\n{boiler(rng, 1800)}\n\n"
             + signature(rng, [(supplier, them_person, signed), (US, us_person, signed)]))
    # forecasts: an earlier one superseded by a later signed one; a draft that changes nothing
    case = rng.choice([6, 12, 24])
    first_fc = rng.randrange(moq // 1000, 150) * 1000 + rng.randrange(0, 1000)
    final_fc = max(moq, first_fc + rng.choice([-1, 1]) * rng.randrange(5, 40) * 1000) // case * case + case
    for day, volume, title in ((signed, first_fc, "Volume forecast 1"),
                               (signed + dt.timedelta(days=rng.randrange(20, 60)), final_fc, "Volume forecast 2")):
        stated = f"{volume:,} units" if volume == first_fc else f"{volume // case:,} cases of {case} units"
        pack.doc("forecast", f"{title} - {part}s", day,
                 f"ANNUAL VOLUME FORECAST for {part}s under the Supply Agreement between {supplier} and {US}\n\n"
                 f"Forecast annual volume: {stated}. This forecast replaces any earlier forecast.\n\n"
                 + signature(rng, [(supplier, them_person, day), (US, us_person, day)]))
    threshold = rng.randrange(max(1, final_fc // 1000 - 30), final_fc // 1000 + 30) * 1000
    threshold += 1000 if threshold == final_fc else 0
    rebate = Decimal(rng.choice(["1", "1.5", "2", "2.5", "3"]))
    rebate_day = signed + dt.timedelta(days=rng.randrange(5, 40))
    pack.doc("side letter", f"Volume rebate letter - {supplier}", rebate_day,
             f"VOLUME REBATE LETTER under the Supply Agreement between {supplier} and {US}\n\nIf the Buyer's purchases "
             f"in a contract year exceed {threshold:,} units, the Supplier rebates {rebate}% of that year's purchase "
             f"value after the year ends.\n\n" + signature(rng, [(supplier, them_person, rebate_day),
                                                                   (US, us_person, rebate_day)]))
    # amendment: a new middle-tier price and maybe a longer term, a new incoterm or indexation
    new_prices = list(prices)
    new_prices[1] = d2(prices[1] - Decimal(rng.randrange(3, 15)) / 100)
    new_term = term + 12 if rng.random() < 0.5 else term
    new_inco = rng.choice([i for i in INCOTERMS if i != inco]) if rng.random() < 0.4 else inco
    new_index = Decimal(rng.choice(["1", "2", "2.75", "4"])) if rng.random() < 0.4 else index_pct
    changes = [f"(a) The unit price of the tier {tiers[1][0]:,} - {tiers[1][1]:,} units becomes {cur} {new_prices[1]}."]
    if new_term != term:
        changes.append(f"(b) The term is extended to {new_term} months.")
    if new_inco != inco:
        changes.append(f"({'bcd'[len(changes) - 1]}) Delivery becomes {new_inco} (Incoterms 2020).")
    if new_index != index_pct:
        changes.append(f"({'bcd'[len(changes) - 1]}) The annual indexation becomes {new_index}%.")
    amend_day = signed + dt.timedelta(days=rng.randrange(60, 200))
    pack.doc("amendment", f"Amendment 1 to the Supply Agreement - {supplier}", amend_day,
             f"AMENDMENT 1 to the Supply Agreement between {supplier} and {US}\n\n" + "\n".join(changes)
             + f"\nAll other terms remain in force.\n\n{boiler(rng, 300)}\n\n"
             + signature(rng, [(supplier, them_person, amend_day), (US, us_person, amend_day)]))
    pack.doc("amendment", f"Amendment 2 to the Supply Agreement - {supplier} (draft)",
             amend_day + dt.timedelta(days=rng.randrange(20, 80)),
             f"AMENDMENT 2 to the Supply Agreement between {supplier} and {US}\n\n{unsigned(supplier)}\n\n(a) All unit "
             f"prices decrease by 8%.\n(b) The Buyer's purchases become exclusive.\n(c) Payment terms become 90 days."
             f"\n\n{boiler(rng, 80)}")
    pack.doc("email", f"RE: {part} forecast", amend_day - dt.timedelta(days=rng.randrange(3, 20)),
             f"From: {us_person}\nTo: {them_person}\n\nWe may need around {final_fc * 2:,} units next year - could "
             f"you hold the bottom tier price for us? Not a commitment, just planning.\n\n{us_person}")
    affiliate_nda(rng, pack, affiliate, signed - dt.timedelta(days=rng.randrange(30, 120)))
    other_contracts(rng, pack, taken, signed)
    price = next(p for (lo, hi), p in zip(tiers, new_prices) if final_fc >= lo and (hi is None or final_fc <= hi))
    pack.field("supplier_legal_name", "name", supplier)
    pack.field("supplier_vat_number", "code", vat)
    pack.field("commencement_date", "date", commence.isoformat())
    pack.field("term_months", "int", new_term)
    pack.field("expiry_date", "date", (add_months(commence, new_term) - dt.timedelta(days=1)).isoformat())
    pack.field("incoterm", "name", new_inco)
    pack.field("currency", "code", cur)
    pack.field("payment_terms_days", "int", pay)
    pack.field("minimum_annual_volume", "int", moq)
    pack.field("forecast_annual_volume", "int", final_fc)
    pack.field("unit_price_at_forecast", "money", float(price))
    pack.field("price_tiers", "table", [{"from_units": lo, "to_units": hi or 0, "unit_price": float(p)}
                                        for (lo, hi), p in zip(tiers, new_prices)])
    pack.field("estimated_annual_value", "money", float(d2(price * final_fc)))
    pack.field("rebate_threshold_units", "int", threshold)
    pack.field("rebate_pct", "number", float(rebate))
    pack.field("estimated_rebate", "money", float(d2(price * final_fc * rebate / 100) if final_fc > threshold else 0))
    pack.field("indexation_pct", "number", float(new_index))
    pack.field("year2_unit_price", "money", float(d2(price * (100 + new_index) / 100)))
    pack.field("late_penalty_pct_per_week", "number", float(penalty))
    pack.field("late_penalty_cap_pct", "number", float(pen_cap))
    pack.field("warranty_months", "int", warranty)
    pack.field("exclusive_supply", "bool", exclusive)
    pack.field("governing_law", "name", law)
    pack.brief = (
        f"Create the deal record for our {part} supply contract with {base}. Deal desk rules: only documents signed "
        f"by both parties count (drafts and emails do not change the deal); a signed amendment prevails over the "
        f"agreement; the latest signed forecast replaces earlier ones. expiry_date is the commencement date plus the "
        f"term (as amended) minus one day. unit_price_at_forecast is the price of the tier the forecast volume falls "
        f"in; estimated_annual_value is forecast volume x that price (forecast volume in units); year2_unit_price "
        f"applies one indexation to it; estimated_rebate is the rebate on the estimated annual value if the forecast "
        f"is above the threshold, else 0; price_tiers lists the tiers as amended; amounts rounded to 2 decimals. Use the supplier's own legal name, not an affiliate's.")
    return pack


# ----------------------------------------------------------------- services --

SERVICES_FIELDS = {
    "supplier_legal_name": ("string", "the supplier's full legal name"),
    "project_name": ("string", "as stated in the statement of work"),
    "start_date": ("string", "YYYY-MM-DD"),
    "end_date": ("string", "YYYY-MM-DD, as amended"),
    "milestones": table("every milestone payable, as amended; name without the M number", "name", name="string",
                        amount="number", due_date="date"),
    "payment_schedule": table("one row per milestone: each is invoiced on its due date", "milestone", milestone="string",
                              pay_by="date", net_payable="number"),
    "total_fixed_fee": ("number", "sum of the milestone amounts"),
    "retention_pct": ("number", "percent withheld until final acceptance"),
    "retention_amount": ("number", "total fixed fee x retention percent"),
    "retention_release_date": ("string", "YYYY-MM-DD"),
    "expenses_cap": ("number", "maximum reimbursable expenses"),
    "day_rate": ("number", "day rate for change requests, as amended"),
    "payment_terms_days": ("integer", "days from invoice date"),
    "acceptance_period_days": ("integer", "days to accept a deliverable"),
    "ip_owner": ("string", "customer or supplier: who owns the deliverables' IP"),
    "key_personnel": ("array", "names of the named key personnel, as amended"),
    "liability_cap": ("number", "the liability cap amount"),
    "governing_law": ("string", "the governing law"),
    "currency": ("string", "ISO code"),
}


def services(rng, taken: set) -> Pack:
    pack = Pack()
    base, supplier, affiliate, _ = parties(rng, taken)
    cur = rng.choice(["GBP", "EUR", "USD"])
    law = rng.choice(LAWS)
    project = f"{synth.word(rng, 2)} {rng.choice(['Migration', 'Data Platform', 'Billing Replatform', 'CRM Rollout'])}"
    start = dt.date(rng.choice([2025, 2026]), rng.randrange(1, 13), rng.randrange(1, 29))
    end = start + dt.timedelta(days=rng.randrange(120, 300))
    names = ["Discovery", "Design sign-off", "Build complete", "User acceptance", "Go-live", "Hypercare exit"]
    count = rng.randrange(3, 5)
    days = sorted(rng.sample(range(20, (end - start).days), count))
    milestones = [{"name": names[i], "amount": float(d2(rng.randrange(80, 900) * 100)),
                   "due_date": (start + dt.timedelta(days=days[i])).isoformat()} for i in range(count)]
    retention = Decimal(rng.choice(["5", "7.5", "10"]))
    day_rate = d2(rng.randrange(600, 1600))
    pay = rng.choice([30, 45, 60])
    accept = rng.choice([5, 10, 15])
    ip_owner = rng.choice(["customer", "supplier"])
    people = {role: synth.person(rng, taken) for role in rng.sample(ROLES, 3)}
    cap_multiple = Decimal(rng.choice(["1", "1.5", "2"]))
    us_person, them_person = synth.person(rng, taken), synth.person(rng, taken)
    signed = start - dt.timedelta(days=rng.randrange(5, 30))
    relative = {i for i in range(count) if rng.random() < 0.5}
    due = lambda i, m: (f"due {days[i]} days after the Start Date" if i in relative  # noqa: E731
                        else f"due {m['due_date']}")
    table = "\n".join(f" M{i + 1} {m['name']:<18} {cur} {m['amount']:>10,.2f}  {due(i, m)}"
                      for i, m in enumerate(milestones))
    release_days = rng.choice([30, 45, 60])
    expenses_pct = Decimal(rng.choice(["5", "8", "10"]))
    pack.doc("statement of work", f"Statement of Work - {project}", signed,
             f"STATEMENT OF WORK under the Services Agreement between {supplier} (the \"Supplier\") and {US} (the "
             f"\"Customer\")\n\nProject: {project}\nStart date: {start.isoformat()}\nPlanned end date: "
             f"{end.isoformat()}\n\n1. Fees. Fixed fee payable by milestone, in {cur}:\n{table}\n"
             f"2. Retention. The Customer withholds {retention}% of every milestone payment and releases it "
             f"{release_days} days after the planned end date.\n"
             f"2A. Expenses. Reasonable expenses are reimbursed up to {expenses_pct}% of the total fixed fee.\n"
             f"3. Changes. Additional work under a signed change request is charged at {cur} {day_rate} per day.\n"
             f"4. Payment. Invoices are payable within {pay} days of the invoice date.\n"
             f"5. Acceptance. The Customer has {accept} business days to accept or reject each deliverable.\n"
             f"6. Intellectual property. "
             + ("All intellectual property in the deliverables vests in the Customer on payment.\n" if ip_owner == "customer"
                else "The Supplier retains all intellectual property and grants the Customer a perpetual licence.\n")
             + "7. Key personnel. " + "; ".join(f"{role}: {name}" for role, name in people.items()) + ".\n"
             f"8. Liability. Each party's liability is capped at {cap_multiple} times the total fixed fee as "
             f"amended from time to time.\n9. Governing law. The laws of {law}.\n\n{boiler(rng, 1400)}\n\n"
             + signature(rng, [(supplier, them_person, signed), (US, us_person, signed)]))
    # change request 1 (signed): a new milestone, a later end date, maybe a new day rate or a replaced person
    extra_name = names[count]
    new_end = end + dt.timedelta(days=rng.randrange(20, 90))
    extra = {"name": extra_name, "amount": float(d2(rng.randrange(40, 400) * 100)),
             "due_date": (new_end - dt.timedelta(days=rng.randrange(1, 15))).isoformat()}
    changes = [f"(a) A new milestone M{count + 1} {extra_name} of {cur} {extra['amount']:,.2f} is added, due "
               f"{extra['due_date']}.", f"(b) The planned end date becomes {new_end.isoformat()}."]
    new_rate = d2(day_rate + rng.randrange(25, 150)) if rng.random() < 0.5 else day_rate
    if new_rate != day_rate:
        changes.append(f"(c) The day rate for change requests becomes {cur} {new_rate}.")
    team = dict(people)
    if rng.random() < 0.6:
        role = rng.choice(list(team))
        team[role] = synth.person(rng, taken)
        changes.append(f"({'cd'[len(changes) - 2]}) {team[role]} replaces {people[role]} as {role}.")
    cr_day = start + dt.timedelta(days=rng.randrange(30, 100))
    pack.doc("change request", f"Change Request CR-1 - {project}", cr_day,
             f"CHANGE REQUEST CR-1 to the Statement of Work for {project}\n\n" + "\n".join(changes)
             + f"\n\n{boiler(rng, 250)}\n\n" + signature(rng, [(supplier, them_person, cr_day), (US, us_person, cr_day)]))
    revised = rng.randrange(count)
    milestones[revised] = dict(milestones[revised], amount=float(d2(milestones[revised]["amount"] * rng.choice([0.9, 1.1, 1.2]))))
    cr3_day = cr_day + dt.timedelta(days=rng.randrange(40, 90))
    pack.doc("change request", f"Change Request CR-3 - {project}", cr3_day,
             f"CHANGE REQUEST CR-3 to the Statement of Work for {project}\n\n(a) The amount of milestone M{revised + 1} "
             f"{milestones[revised]['name']} becomes {cur} {milestones[revised]['amount']:,.2f}.\nAll other terms of the "
             f"Statement of Work as amended by CR-1 remain unchanged.\n\n{boiler(rng, 150)}\n\n"
             + signature(rng, [(supplier, them_person, cr3_day), (US, us_person, cr3_day)]))
    pack.doc("change request", f"Change Request CR-2 - {project} (draft)", cr_day + dt.timedelta(days=rng.randrange(10, 40)),
             f"CHANGE REQUEST CR-2 to the Statement of Work for {project}\n\n{unsigned(supplier)}\n\n(a) Milestone "
             f"M1 increases by {cur} 25,000.00.\n(b) The retention is removed.\n\n{boiler(rng, 80)}")
    pack.doc("email", f"RE: {project} day rate", cr_day - dt.timedelta(days=rng.randrange(3, 15)),
             f"From: {them_person}\nTo: {us_person}\n\nWe could drop the day rate to {cur} {day_rate - 100} if CR-2 goes "
             f"ahead. Subject to contract.\n\n{them_person}")
    affiliate_nda(rng, pack, affiliate, signed - dt.timedelta(days=rng.randrange(30, 120)))
    other_contracts(rng, pack, taken, signed)
    rows = milestones + [extra]
    total = d2(sum(Decimal(str(m["amount"])) for m in rows))
    pack.field("supplier_legal_name", "name", supplier)
    pack.field("project_name", "name", project)
    pack.field("start_date", "date", start.isoformat())
    pack.field("end_date", "date", new_end.isoformat())
    pack.field("milestones", "table", rows)
    pack.field("payment_schedule", "table", [
        {"milestone": m["name"], "pay_by": (dt.date.fromisoformat(m["due_date"]) + dt.timedelta(days=pay)).isoformat(),
         "net_payable": float(d2(Decimal(str(m["amount"])) * (100 - retention) / 100))} for m in rows])
    pack.field("total_fixed_fee", "money", float(total))
    pack.field("retention_pct", "number", float(retention))
    pack.field("retention_amount", "money", float(d2(total * retention / 100)))
    pack.field("retention_release_date", "date", (new_end + dt.timedelta(days=release_days)).isoformat())
    pack.field("expenses_cap", "money", float(d2(total * expenses_pct / 100)))
    pack.field("day_rate", "money", float(new_rate))
    pack.field("payment_terms_days", "int", pay)
    pack.field("acceptance_period_days", "int", accept)
    pack.field("ip_owner", "name", ip_owner)
    pack.field("key_personnel", "names", sorted(team.values()))
    pack.field("liability_cap", "money", float(d2(total * cap_multiple)))
    pack.field("governing_law", "name", law)
    pack.field("currency", "code", cur)
    pack.brief = (
        f"Create the deal record for our {project} services engagement with {base}. Deal desk rules: only documents "
        f"signed by both parties count (drafts and emails do not change the deal); signed change requests amend the "
        f"statement of work in order. Milestones list every milestone payable as amended (name as written without the "
        f"M number, amount, due date as YYYY-MM-DD); payment_schedule gives for each milestone the date payment is "
        f"due (its due date plus the payment terms) and the amount net of retention; retention_amount is "
        f"total_fixed_fee x retention_pct / 100, the "
        f"release date counts from the planned end date as amended, and liability_cap and expenses_cap are the stated "
        f"multiple and percentage of the total fixed fee, all rounded to 2 decimals. Use the supplier's own legal name, not "
        f"an affiliate's.")
    return pack


JSON = {"string": "string", "date": "string", "integer": "integer", "number": "number"}


def submit_tool(name: str, title: str, fields: dict) -> dict:
    props = {}
    for key, spec in fields.items():
        kind, desc = spec[0], spec[1]
        if kind == "table":
            props[key] = {"type": "array", "description": desc, "items": {
                "type": "object", "properties": {c: {"type": JSON[t]} for c, t in spec[3].items()},
                "required": list(spec[3]), "additionalProperties": False}}
        else:
            props[key] = {"type": kind, "description": desc}
    return tool(name, f"Submit the {title} deal record (ends the task).", props, list(fields))


KINDS = {"saas": (saas, SAAS_FIELDS, "submit_saas_deal", "software subscription"),
         "supply": (supply, SUPPLY_FIELDS, "submit_supply_deal", "supply"),
         "services": (services, SERVICES_FIELDS, "submit_services_deal", "services")}
