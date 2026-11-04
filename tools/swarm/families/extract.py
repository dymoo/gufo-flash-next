"""Extraction agent: find business documents and submit a structured record.

Tiers: an invoice with labelled fields (easy); a corrected invoice that replaces
an earlier one under the same number, with net-terms due date, a discount and
mixed number formats (medium); an incident review in a local time zone whose
record needs UTC conversion, time arithmetic and only the customer-impacting
services, or a purchase order reconciled against shipment notices with a
cancelled notice, a goods return and other orders' notices (hard). Records are
submitted through strict-schema tools and compared field by field.
"""

from __future__ import annotations

import datetime as dt
import re

from . import Call, Step, Task, ToolError, rng_for, synth, tool

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
CURRENCIES = {"EUR": ("€", 0.19), "USD": ("$", 0.0), "GBP": ("£", 0.20), "CHF": ("CHF", 0.081), "JPY": ("¥", 0.10)}
PRODUCTS = ["fastener kit M6", "hydraulic seal set", "LED panel 60x60", "safety gloves (pair)", "cable tray 2 m",
            "label printer ribbon", "pallet wrap roll", "torque wrench 1/2in", "air filter cartridge", "copper bus bar",
            "thermal paste 50 g", "bearing 6204-2RS", "solenoid valve 24 V", "nitrile hose 10 m", "packing foam sheet",
            "barcode scanner", "conveyor belt link", "stainless bolt M10", "fuse 16 A", "insulation tape"]
SERVICES = ["checkout-api", "payments-gateway", "search", "auth-service", "notifications", "inventory-sync",
            "media-cdn", "reporting", "order-history", "pricing-engine"]
ROOT_CAUSES = {
    "deployment": "A release of {svc} shipped a change that regressed request handling; rolling back the release ended the impact.",
    "configuration": "A manual configuration change to {svc} set a timeout far below the normal value; restoring the previous configuration ended the impact.",
    "capacity": "Traffic to {svc} exceeded the provisioned connection pool and requests queued until they timed out; adding capacity ended the impact.",
    "dependency": "An upstream provider that {svc} calls returned errors; failing over to the secondary provider ended the impact.",
    "hardware": "A storage node behind {svc} suffered a disk failure; evacuating the node ended the impact.",
    "security": "An expired TLS certificate on {svc} made clients reject connections; issuing a new certificate ended the impact.",
}


def fmt_date(d: dt.date, style: int) -> str:
    return [f"{d.day} {MONTHS[d.month - 1]} {d.year}", f"{MONTHS[d.month - 1]} {d.day}, {d.year}", d.isoformat(),
            f"{d.day:02d}.{d.month:02d}.{d.year}"][style]


def fmt_amount(x: float, currency: str, european: bool) -> str:
    text = f"{x:,.0f}" if currency == "JPY" else f"{x:,.2f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".") if european else text


def norm_name(text: str) -> str:
    text = re.sub(r"[^a-z0-9 ]", " ", text.lower())
    return " ".join(w for w in text.split() if w not in {"gmbh", "ltd", "inc", "s", "a", "b", "v", "ab", "oy", "r", "l"})


def money(x) -> float:
    return round(float(x), 2)


SUBMIT_INVOICE = tool(
    "submit_invoice", "Submit the extracted invoice record (ends the task).",
    {"vendor": {"type": "string", "description": "Issuer's legal name"},
     "invoice_number": {"type": "string"},
     "issue_date": {"type": "string", "description": "YYYY-MM-DD"},
     "due_date": {"type": "string", "description": "YYYY-MM-DD"},
     "currency": {"type": "string", "enum": list(CURRENCIES)},
     "total": {"type": "number", "description": "Grand total payable, including tax"},
     "line_items": {"type": "array", "items": {"type": "object", "properties": {
         "sku": {"type": "string"}, "quantity": {"type": "integer"}, "unit_price": {"type": "number"}},
         "required": ["sku", "quantity", "unit_price"], "additionalProperties": False}}},
    ["vendor", "invoice_number", "issue_date", "due_date", "currency", "total", "line_items"])
SUBMIT_INCIDENT = tool(
    "submit_incident", "Submit the structured incident record (ends the task).",
    {"incident_id": {"type": "string"},
     "severity": {"type": "string", "enum": ["SEV1", "SEV2", "SEV3", "SEV4"]},
     "started_at": {"type": "string", "description": "First customer impact, UTC, YYYY-MM-DDTHH:MM"},
     "resolved_at": {"type": "string", "description": "When the incident was declared resolved, UTC, YYYY-MM-DDTHH:MM"},
     "duration_minutes": {"type": "integer"},
     "services": {"type": "array", "items": {"type": "string"}, "description": "Services with customer impact"},
     "root_cause": {"type": "string", "enum": list(ROOT_CAUSES)}},
    ["incident_id", "severity", "started_at", "resolved_at", "duration_minutes", "services", "root_cause"])
SUBMIT_RECON = tool(
    "submit_reconciliation", "Submit the purchase-order reconciliation (ends the task).",
    {"po_number": {"type": "string"}, "vendor": {"type": "string"},
     "ordered_units": {"type": "integer"}, "shipped_units": {"type": "integer"},
     "backordered": {"type": "array", "items": {"type": "object", "properties": {
         "sku": {"type": "string"}, "missing_quantity": {"type": "integer"}},
         "required": ["sku", "missing_quantity"], "additionalProperties": False}}},
    ["po_number", "vendor", "ordered_units", "shipped_units", "backordered"])


class ExtractTask(Task):
    family = "extract"
    max_turns = 20
    tools = [
        tool("search_documents", "Full-text search over the document archive. Returns up to 8 matches as "
             "doc id | type | title | date.", {"query": {"type": "string"}}, ["query"]),
        tool("get_document", "Return the full text of a document.", {"doc_id": {"type": "string"}}, ["doc_id"]),
        SUBMIT_INVOICE, SUBMIT_INCIDENT, SUBMIT_RECON,
    ]
    latency = {"search_documents": 1.0, "get_document": 1.0}

    def __init__(self, index: int, seed: int):
        super().__init__(index, seed)
        self.rng = rng = rng_for(self.family, seed, index)
        self.docs: dict[str, dict] = {}  # id -> {type, title, date, text}
        self.names: set = set()
        kind = ["invoice", "corrected", "incident", "reconcile"][index % 4]
        self.kind = kind
        self.tier = {"invoice": "easy", "corrected": "medium"}.get(kind, "hard")
        header = self.header()
        if kind in ("invoice", "corrected"):
            hard = kind == "corrected"
            vendor = synth.company(rng, self.names) + " " + rng.choice(synth.LEGAL)
            self.expected, doc_ids = self._invoice(vendor, hard, target=True)
            if hard:  # the vendor later issued a corrected invoice under the same number
                self.expected, doc_ids = self._invoice(vendor, hard, replaces=self.expected)
            for _ in range(rng.randrange(3, 6)):
                self._invoice(synth.company(rng, self.names) + " " + rng.choice(synth.LEGAL), rng.random() < 0.5)
            self._invoice(vendor, hard)  # another invoice from the same vendor
            self.submit_tool = "submit_invoice"
            short = vendor.rsplit(" ", 1)[0]
            self.prompt = (f"{header} Accounts payable needs invoice {self.expected['invoice_number']} from {short} "
                           f"entered into the ledger. Find it with search_documents, read it, and submit it with "
                           f"submit_invoice. Dates are ISO (YYYY-MM-DD); total is the grand total payable including "
                           f"tax; line_items are the product lines only (no shipping, discount or tax lines). If the "
                           f"vendor issued a correction, the corrected invoice is the one to record.")
        elif kind == "incident":
            self.expected = self._incident(target=True)
            for _ in range(rng.randrange(3, 6)):
                self._incident()
            self.submit_tool = "submit_incident"
            self.prompt = (f"{header} File the structured record for incident {self.expected['incident_id']}: find its "
                           f"post-incident review with search_documents, read it, then call submit_incident. "
                           f"started_at is the first customer-impact time and resolved_at the time the incident was "
                           f"declared resolved, both converted to UTC (YYYY-MM-DDTHH:MM); duration_minutes is the "
                           f"difference; services lists only the services that had customer impact.")
        else:
            self.expected = self._reconcile()
            self.submit_tool = "submit_reconciliation"
            self.prompt = (f"{header} Reconcile purchase order {self.expected['po_number']} against its shipment "
                           f"notices and submit the result with submit_reconciliation. Ignore cancelled notices and "
                           f"notices for other orders; units the warehouse returned to the supplier count as not "
                           f"received. ordered_units and shipped_units (received and kept) are totals over all lines; "
                           f"backordered lists every SKU with units still missing, sorted by SKU.")

    # ------------------------------------------------------------ documents --

    def _add(self, kind: str, title: str, date: str, text: str) -> str:
        while True:
            doc_id = f"DOC-{self.rng.randrange(10000, 99999)}"
            if doc_id not in self.docs:
                break
        self.docs[doc_id] = {"type": kind, "title": title, "date": date, "text": text}
        return doc_id

    def _letterhead(self, vendor: str) -> str:
        rng = self.rng
        domain = re.sub(r"[^a-z]", "", vendor.lower().split()[0]) + ".example"
        return (f"{vendor.upper()}\n{synth.street(rng)} · {rng.randrange(10000, 99999)} "
                f"{synth.city(rng, set())}\nVAT ID: EU{rng.randrange(10**8, 10**9)} · Registered office: "
                f"{synth.city(rng, set())} · Phone +{rng.randrange(10, 99)} {rng.randrange(100, 999)} "
                f"{rng.randrange(1000, 9999)} · accounts@{domain}\n")

    def _invoice(self, vendor: str, hard: bool, target: bool = False, replaces: dict | None = None):
        rng = self.rng
        currency = replaces["currency"] if replaces else rng.choice(list(CURRENCIES))
        symbol, rate = CURRENCIES[currency]
        european = hard and currency in ("EUR", "CHF") and rng.random() < 0.6
        style = rng.randrange(4) if hard else rng.choice([0, 2])
        issue = (dt.date.fromisoformat(replaces["issue_date"]) + dt.timedelta(days=rng.randrange(3, 12)) if replaces
                 else dt.date(2026, 1, 1) + dt.timedelta(days=rng.randrange(0, 260)))
        terms = rng.choice([14, 30, 45, 60])
        due = issue + dt.timedelta(days=terms)
        number = replaces["invoice_number"] if replaces else f"INV-2026-{rng.randrange(10000, 99999)}"
        items = []
        if replaces:  # same lines, one quantity and one price corrected
            items = [dict(i, description=rng.choice(PRODUCTS)) for i in replaces["line_items"]]
            items[rng.randrange(len(items))]["quantity"] += rng.choice([-1, 1, 2, 5])
            items[rng.randrange(len(items))]["unit_price"] = (rng.randrange(100, 90000) if currency == "JPY"
                                                               else rng.randrange(150, 90000) / 100)
            for i in items:
                i["quantity"] = max(1, i["quantity"])
        for sku_prefix in [] if replaces else rng.sample(["AX", "BK", "CR", "DV", "EL", "FM", "GT", "HR", "KL", "MP"],
                                                          rng.randrange(2, 7)):
            price = rng.randrange(100, 90000) if currency == "JPY" else rng.randrange(150, 90000) / 100
            items.append({"sku": f"{sku_prefix}-{rng.randrange(1000, 9999)}", "quantity": rng.randrange(1, 41),
                          "unit_price": price, "description": rng.choice(PRODUCTS)})
        subtotal = sum(i["quantity"] * i["unit_price"] for i in items)
        discount_pct = rng.choice([2, 3, 5, 10]) if hard else 0
        discount = round(subtotal * discount_pct / 100, 0 if currency == "JPY" else 2)
        shipping = rng.choice([0, 0, rng.randrange(500, 6000) / 100]) if currency != "JPY" else rng.choice([0, 1500])
        tax = round((subtotal - discount + shipping) * rate, 0 if currency == "JPY" else 2)
        total = round(subtotal - discount + shipping + tax, 0 if currency == "JPY" else 2)
        amount = lambda x: fmt_amount(x, currency, european)  # noqa: E731
        order_date = issue - dt.timedelta(days=rng.randrange(5, 25))
        delivery = issue - dt.timedelta(days=rng.randrange(1, 5))
        lines = "\n".join(f" {n:>3} | {i['sku']:<8} | {i['description']:<22} | {i['quantity']:>4} | "
                          f"{amount(i['unit_price']):>10} | {amount(i['quantity'] * i['unit_price']):>12}"
                          for n, i in enumerate(items, 1))
        due_line = (f"Payment terms: net {terms} days from the invoice date" if hard else
                    f"Due date: {fmt_date(due, style)} (net {terms} days)")
        totals = [f"Subtotal: {currency} {amount(subtotal)}"]
        if discount:
            totals.append(f"Early-order discount ({discount_pct}%): -{currency} {amount(discount)}")
        if shipping:
            totals.append(f"Shipping and handling: {currency} {amount(shipping)}")
        totals.append(f"Tax ({rate * 100:g}%): {currency} {amount(tax)}")
        totals.append(f"TOTAL DUE: {symbol} {amount(total)} ({currency})")
        previous = rng.randrange(1000, 500000) / 100
        banner = (f"CORRECTED INVOICE\nThis invoice replaces invoice {number} issued on "
                  f"{fmt_date(dt.date.fromisoformat(replaces['issue_date']), style)}; that invoice is void.\n"
                  if replaces else "INVOICE\n")
        text = (self._letterhead(vendor) + "\n" + banner + "\n"
                f"Invoice number: {number}\nInvoice date: {fmt_date(issue, style)}\n"
                f"Customer number: C-{rng.randrange(10000, 99999)}\n"
                f"Your order: PO-{rng.randrange(10000, 99999)} of {fmt_date(order_date, style)}\n"
                f"Delivery date: {fmt_date(delivery, style)}\n{due_line}\n\n"
                f"Bill to: Northwind Analytics Ltd, Accounts Payable, {synth.street(rng)}, Leeds\n\n"
                f" Pos | SKU      | Description            |  Qty | Unit price | Amount\n{lines}\n\n"
                + "\n".join(totals) + "\n\n"
                f"Statement summary: previous balance {currency} {amount(previous)}; payments received "
                f"{currency} {amount(previous)}; balance brought forward {currency} {amount(0)}.\n"
                + (f"Amounts use a decimal comma.\n" if european else "") +
                f"\nBank: IBAN DE{rng.randrange(10**19, 10**20)} · BIC {synth.word(rng, 2).upper()[:6]}XXX\n\n"
                f"General terms and conditions of sale\n{synth.boilerplate(rng, rng.randrange(350, 800))}\n")
        doc_id = self._add("invoice", f"{vendor} — {number}" + (" (corrected)" if replaces else ""),
                           issue.isoformat(), text)
        record = {"vendor": vendor, "invoice_number": number, "issue_date": issue.isoformat(),
                  "due_date": due.isoformat(), "currency": currency, "total": total,
                  "line_items": [{k: i[k] for k in ("sku", "quantity", "unit_price")} for i in items]}
        return record, [doc_id]

    def _incident(self, target: bool = False) -> dict:
        rng = self.rng
        inc = f"INC-{rng.randrange(1000, 9999)}"
        start = dt.datetime(2026, 1, 1, 0, 0) + dt.timedelta(minutes=rng.randrange(0, 260 * 1440))
        zone, offset = rng.choice([("CET (UTC+1)", 60), ("CEST (UTC+2)", 120), ("EST (UTC-5)", -300),
                                   ("PDT (UTC-7)", -420), ("IST (UTC+5:30)", 330), ("JST (UTC+9)", 540)])
        detect = start + dt.timedelta(minutes=rng.randrange(2, 16))
        ack = detect + dt.timedelta(minutes=rng.randrange(2, 12))
        mitigate = ack + dt.timedelta(minutes=rng.randrange(15, 240))
        resolved = mitigate + dt.timedelta(minutes=rng.randrange(10, 180))
        severity = rng.choice(["SEV1", "SEV2", "SEV3", "SEV4"])
        services = rng.sample(SERVICES, rng.randrange(1, 4))
        unaffected = rng.sample([s for s in SERVICES if s not in services], 2)
        cause = rng.choice(list(ROOT_CAUSES))
        owner = synth.person(rng, self.names)

        def stamp(t: dt.datetime, prev: dt.datetime | None) -> str:
            t, prev = t + dt.timedelta(minutes=offset), prev and prev + dt.timedelta(minutes=offset)
            return t.strftime("%H:%M") if prev and prev.date() == t.date() else t.strftime("%Y-%m-%d %H:%M")
        events = [(detect - dt.timedelta(minutes=rng.randrange(20, 90)), "Routine deploy window opens; no changes to affected systems."),
                  (start, f"Impact begins: error rate on {services[0]} rises and customers see failures."),
                  (detect, "Alert fires for elevated 5xx responses."),
                  (ack, f"On-call engineer {owner} acknowledges the page."),
                  (ack + dt.timedelta(minutes=5), f"Investigated {unaffected[0]} and {unaffected[1]}; both healthy, not affected."),
                  (mitigate, "Mitigation applied; error rates start to fall."),
                  (resolved, "Metrics back at baseline for 30 minutes; incident declared resolved."),
                  (resolved + dt.timedelta(minutes=rng.randrange(30, 300)), "Post-incident review scheduled.")]
        lines, prev = [], None
        for t, what in sorted(events, key=lambda e: e[0]):
            lines.append(f"- {stamp(t, prev)} — {what}")
            prev = t
        impact = ", ".join(services[:-1]) + (" and " if len(services) > 1 else "") + services[-1]
        internal = rng.choice([s for s in SERVICES if s not in services and s not in unaffected])
        text = (f"POST-INCIDENT REVIEW — {inc}\n\nSeverity: {severity}\nAuthor: {owner}\n"
                f"Status: closed\n\nSummary\n{ROOT_CAUSES[cause].format(svc=services[0])}\n\n"
                f"Customer impact\nCustomers saw errors from {impact}. {internal} was degraded for internal "
                f"batch jobs only; no customer-facing request was affected.\n\n"
                f"Timeline (all times {zone}; the date is shown when it changes)\n" + "\n".join(lines) + "\n\n"
                f"Root cause\n{ROOT_CAUSES[cause].format(svc=services[0])}\n\nFollow-up actions\n"
                + synth.boilerplate(rng, rng.randrange(250, 600)) + "\n")
        self._add("incident review", f"Post-incident review {inc}", start.date().isoformat(), text)
        return {"incident_id": inc, "severity": severity, "started_at": start.strftime("%Y-%m-%dT%H:%M"),
                "resolved_at": resolved.strftime("%Y-%m-%dT%H:%M"),
                "duration_minutes": int((resolved - start).total_seconds() // 60),
                "services": sorted(services), "root_cause": cause}

    def _reconcile(self) -> dict:
        rng = self.rng
        vendor = synth.company(rng, self.names) + " " + rng.choice(synth.LEGAL)
        po = f"PO-{rng.randrange(10000, 99999)}"
        order_date = dt.date(2026, 1, 1) + dt.timedelta(days=rng.randrange(0, 200))
        skus = sorted(f"{p}-{rng.randrange(1000, 9999)}" for p in rng.sample(["AX", "BK", "CR", "DV", "EL", "FM", "GT"],
                                                                             rng.randrange(3, 7)))
        ordered = {s: rng.randrange(5, 201) for s in skus}
        shipped = {s: 0 for s in skus}
        n_notices = rng.randrange(2, 4)
        notices = [dict() for _ in range(n_notices)]
        full = rng.choice(skus)
        missing = rng.choice([s for s in skus if s != full])
        for s in skus:
            total = ordered[s] if s == full else (rng.randrange(0, ordered[s]) if s == missing
                                                  else rng.choice([ordered[s], rng.randrange(0, ordered[s] + 1)]))
            for k in range(n_notices):  # split the shipped amount over the notices
                part = total if k == n_notices - 1 else rng.randrange(0, total + 1)
                if part:
                    notices[k][s] = notices[k].get(s, 0) + part
                total -= part
            shipped[s] = sum(n.get(s, 0) for n in notices)
        lines = "\n".join(f" {s:<8} | {rng.choice(PRODUCTS):<22} | {ordered[s]:>5}" for s in skus)
        self._add("purchase order", f"Purchase order {po} — {vendor}", order_date.isoformat(),
                  f"NORTHWIND ANALYTICS LTD — PURCHASE ORDER\n\nPO number: {po}\nSupplier: {vendor}\n"
                  f"Order date: {order_date.isoformat()}\nDelivery address: {synth.street(rng)}, Leeds\n\n"
                  f" SKU      | Description            | Qty ordered\n{lines}\n\n"
                  f"Terms\n{synth.boilerplate(rng, rng.randrange(250, 600))}\n")

        def notice(ref: str, items: dict, status: str, day: int) -> None:
            sn = f"SN-{rng.randrange(100000, 999999)}"
            date = order_date + dt.timedelta(days=day)
            body = "\n".join(f" {s:<8} | {q:>5}" for s, q in sorted(items.items())) or " (no items)"
            self._add("shipment notice", f"Shipment notice {sn} — {vendor}", date.isoformat(),
                      self._letterhead(vendor) + f"\nSHIPMENT NOTICE {sn}\n\nCustomer order reference: {ref}\n"
                      f"Status: {status}\nDispatch date: {date.isoformat()}\nCarrier: {synth.word(rng, 2)} Freight, "
                      f"tracking {rng.randrange(10**11, 10**12)}\n\n SKU      | Qty shipped\n{body}\n\n"
                      f"Notes\n{synth.boilerplate(rng, rng.randrange(150, 400))}\n")
        for k, items in enumerate(notices):
            notice(po, items, "DISPATCHED", 7 + 5 * k)
        returned = {}
        kept = [s for s in skus if shipped[s] > 0]
        if kept:  # damaged units sent back to the supplier
            s = rng.choice(kept)
            returned[s] = rng.randrange(1, shipped[s] + 1)
            date = order_date + dt.timedelta(days=7 + 5 * n_notices + rng.randrange(1, 6))
            self._add("return notice", f"Return to supplier RMA-{rng.randrange(10000, 99999)} — {vendor}", date.isoformat(),
                      f"NORTHWIND ANALYTICS LTD — GOODS RETURN NOTE\n\nSupplier: {vendor}\nCustomer order reference: "
                      f"{po}\nReason: damaged in transit; returned to the supplier for credit, not to be replaced.\n\n"
                      f" SKU      | Qty returned\n {s:<8} | {returned[s]:>5}\n\n"
                      f"Notes\n{synth.boilerplate(rng, rng.randrange(80, 200))}\n")
            shipped[s] -= returned[s]
        if rng.random() < 0.6:
            notice(po, {s: rng.randrange(1, 30) for s in rng.sample(skus, 2)}, "CANCELLED — not dispatched", 6)
        for _ in range(rng.randrange(1, 3)):
            other = f"PO-{rng.randrange(10000, 99999)}"
            notice(other, {s: rng.randrange(1, 50) for s in rng.sample(skus, 2)}, "DISPATCHED", rng.randrange(3, 30))
        backordered = [{"sku": s, "missing_quantity": ordered[s] - shipped[s]} for s in skus if ordered[s] > shipped[s]]
        return {"po_number": po, "vendor": vendor, "ordered_units": sum(ordered.values()),
                "shipped_units": sum(shipped.values()), "backordered": backordered}

    # ---------------------------------------------------------------- tools --

    def tool_search_documents(self, query: str) -> str:
        terms = [t for t in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", query.lower()) if len(t) > 1]
        if not terms:
            raise ToolError("the query has no searchable terms")
        scored = []
        for doc_id, d in self.docs.items():
            hay = (d["title"] + " " + d["text"]).lower()
            words = set(re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", hay))
            score = sum(2 if t in d["title"].lower() else 1 for t in terms if t in words)
            if score:
                scored.append((-score, doc_id))
        if not scored:
            return "no matches"
        return "\n".join(f"{i} | {self.docs[i]['type']} | {self.docs[i]['title']} | {self.docs[i]['date']}"
                         for _, i in sorted(scored)[:8])

    def tool_get_document(self, doc_id: str) -> str:
        doc = self.docs.get(doc_id.strip().upper())
        if doc is None:
            raise ToolError(f"no document {doc_id!r}")
        return doc["text"]

    # -------------------------------------------------------------- checker --

    def call(self, name: str, arguments: dict) -> str:
        if name.startswith("submit_") and name != self.submit_tool:
            raise ToolError(f"this ticket must be submitted with {self.submit_tool}")
        return super().call(name, arguments)

    def check(self, final: str) -> tuple[bool, str]:
        got, want = self.submitted, self.expected
        if got is None:
            return False, "no submission"
        wrong = []
        same_text = lambda a, b: str(a).strip().lower() == str(b).strip().lower()  # noqa: E731
        for key, value in want.items():
            have = got.get(key)
            if key == "vendor":
                ok = norm_name(str(have)) == norm_name(value)
            elif key == "total":
                ok = isinstance(have, (int, float)) and abs(have - value) < 0.011
            elif key == "line_items":
                ok = isinstance(have, list) and sorted((str(i.get("sku", "")).upper(), i.get("quantity"),
                                                        money(i.get("unit_price", 0))) for i in have) == \
                    sorted((i["sku"], i["quantity"], money(i["unit_price"])) for i in value)
            elif key == "services":
                ok = isinstance(have, list) and sorted(str(s).strip().lower() for s in have) == value
            elif key == "backordered":
                ok = isinstance(have, list) and [(str(i.get("sku", "")).upper(), i.get("missing_quantity"))
                                                 for i in have] == [(i["sku"], i["missing_quantity"]) for i in value]
            else:
                ok = same_text(have, value)
            if not ok:
                wrong.append(key)
        return not wrong, "all fields match" if not wrong else "wrong: " + ", ".join(wrong)

    def oracle(self) -> list[Step]:
        if self.kind == "reconcile":
            query = self.expected["po_number"]
        elif self.kind == "incident":
            query = self.expected["incident_id"]
        else:
            query = self.expected["invoice_number"]
        wanted = [i for i, d in self.docs.items() if query in d["text"] and "CANCELLED" not in d["text"]]
        return [Step(calls=[Call("search_documents", {"query": query})]),
                Step(calls=[Call("get_document", {"doc_id": i}) for i in sorted(wanted)]),
                Step(calls=[Call(self.submit_tool, dict(self.expected))]),
                Step(content="Submitted.")]
