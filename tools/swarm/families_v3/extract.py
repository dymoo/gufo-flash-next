"""Extraction (hard tier): a deal desk's briefs over document packs, every field one requirement.

Variants: three DealGate-style deal records - a software subscription (master
agreement, order form, amendment, side letter), a supply agreement (tiered prices,
forecasts, indexation) and a services statement of work (milestones, change
requests, retention) - each with documents that must not change the record
(unsigned drafts, "subject to contract" emails, a rate card, an affiliate's NDA);
and an accounts-payable open-items ledger for one supplier: invoices split over two
pages or re-sent as copies, superseding amendments, a cancellation, credit notes,
remittances that pay several invoices (some in GBP against EUR/USD invoices at the
payment day's rate), dated exchange rates, a sister company with a near-identical
name and OCR noise. Each deal field, and each kind of ledger item, is reported as a
requirement.
"""

from __future__ import annotations

import datetime as dt
import re

from families import Call, Step, Task, ToolError, rng_for, synth, tool

from . import deals
from .rows import rows_match

PRODUCTS = ["fastener kit M6", "hydraulic seal set", "LED panel 60x60", "safety gloves (pair)", "cable tray 2 m",
            "label printer ribbon", "pallet wrap roll", "torque wrench 1/2in", "air filter cartridge", "copper bus bar",
            "thermal paste 50 g", "bearing 6204-2RS", "solenoid valve 24 V", "nitrile hose 10 m", "fuse 16 A"]
BUCKETS = ["current", "1-30", "31-60", "61+"]
SUFFIXES = [("GmbH", "AG"), ("Ltd", "Limited"), ("S.A.", "S.A.S."), ("B.V.", "N.V."), ("Inc.", "LLC")]
OCR = [("rn", "m"), ("l", "1"), ("O", "0"), ("cl", "d"), ("e", "c")]


def noisy(rng, text: str) -> str:
    """OCR-style character confusions in roughly one word in twelve (never digits, never key fields)."""
    words = text.split(" ")
    for i, w in enumerate(words):
        if w.isalpha() and len(w) > 4 and rng.random() < 0.08:
            a, b = rng.choice(OCR)
            words[i] = w.replace(a, b, 1)
    return " ".join(words)


def money(x: float) -> float:
    return round(x + 1e-9, 2)


def bucket(days: int) -> str:
    return "current" if days <= 0 else "1-30" if days <= 30 else "31-60" if days <= 60 else "61+"


SUBMIT = tool(
    "submit_ledger", "Submit the reconciled ledger record (ends the task).",
    {"vendor": {"type": "string", "description": "The vendor's legal name"},
     "as_of": {"type": "string", "description": "YYYY-MM-DD"},
     "invoices": {"type": "array", "description": "Every invoice of the vendor with an outstanding balance", "items": {
         "type": "object", "properties": {
             "invoice_number": {"type": "string"},
             "currency": {"type": "string", "enum": ["GBP", "EUR", "USD"]},
             "outstanding": {"type": "number", "description": "In the invoice currency"},
             "outstanding_gbp": {"type": "number", "description": "Converted at the rate dated on the invoice date"},
             "bucket": {"type": "string", "enum": BUCKETS}},
         "required": ["invoice_number", "currency", "outstanding", "outstanding_gbp", "bucket"],
         "additionalProperties": False}},
     "total_outstanding_gbp": {"type": "number"}},
    ["vendor", "as_of", "invoices", "total_outstanding_gbp"])


class Extract3Task(Task):
    family = "extract3"
    max_turns = 40
    submit_tool = "submit_ledger"
    tools = [
        tool("search_documents", "Full-text search over the document archive, best matches first, 6 per page, "
             "as doc id | type | title | date.",
             {"query": {"type": "string"}, "page": {"type": "integer", "description": "1-based, default 1"}},
             ["query"]),
        tool("get_document", "Return the full text of a document.", {"doc_id": {"type": "string"}}, ["doc_id"]),
        SUBMIT,
        *(deals.submit_tool(name, title, fields) for _, fields, name, title in deals.KINDS.values()),
    ]
    latency = {"search_documents": 1.0, "get_document": 1.0}
    VARIANTS = ["saas", "supply", "services", "ledger"]

    def __init__(self, index: int, seed: int):
        super().__init__(index, seed)
        self.rng = rng = rng_for(self.family, seed, index)
        self.tier = "hard"
        self.kind = self.VARIANTS[index % 4]
        self.docs: dict[str, dict] = {}
        if self.kind == "ledger":
            self._ledger(rng)
            return
        make, self.fields, self.submit_tool, _ = deals.KINDS[self.kind]
        pack = make(rng, set())
        for kind, title, date, text in pack.docs:
            self._add(kind, title, date, text)
        self.expected, self.types = pack.expected, pack.types
        self.prompt = (f"{self.header()} {pack.brief} Find the documents with search_documents and get_document "
                       f"(several documents concern other parties or are not binding), then submit the record with "
                       f"{self.submit_tool}. Dates are YYYY-MM-DD; amounts are plain numbers in the deal currency.")

    def _ledger(self, rng) -> None:
        variant = 3
        taken: set = set()
        base = synth.company(rng, taken)
        suffix, sister_suffix = rng.choice(SUFFIXES)
        self.vendor, self.sister = f"{base} {suffix}", f"{base} {sister_suffix}"
        self.vat = f"EU{rng.randrange(10**8, 10**9)}"
        self.numbers: set = set()
        self.letterheads: dict[str, str] = {}
        self.as_of = dt.date(2026, 9, 1) + dt.timedelta(days=rng.randrange(0, 45))
        self.rates = self._rates(rng)
        self.inv_tags: dict[str, set] = {}
        n_invoices = 10 + rng.randrange(0, 4)
        invoices = [self._invoice(rng, self.vendor, multipage=(i % 2 == 0)) for i in range(n_invoices)]
        for i, inv in enumerate(invoices):
            self.inv_tags[inv["number"]] = {"split_over_two_pages"} if i % 2 == 0 else set()
        modes = [rng.choice(["full", "partial", "partial", "none"]) for _ in invoices]
        if sum(m != "full" for m in modes) < 3:  # at least three open items
            modes[0], modes[1], modes[2] = "partial", "none", "partial"
        cancelled = {rng.randrange(3, n_invoices)} if variant >= 2 else set()
        parts = []  # (invoice, amount in the invoice currency)
        for i, (inv, mode) in enumerate(zip(invoices, modes)):
            tags = self.inv_tags[inv["number"]]
            if rng.random() < 0.3:
                self._copy(rng, inv)
                tags.add("resent_copy")
            if i in cancelled:
                self._cancel(rng, inv)
                tags.add("cancelled_excluded")
                inv["amount"] = inv["credited"] = 0.0
                continue
            amount = inv["total"]
            if variant >= 1 and rng.random() < 0.5:  # amended total, later date
                amount = self._amendment(rng, inv)
                tags.add("amended_total")
            credited = self._credit(rng, inv, amount) if rng.random() < 0.4 else 0.0
            if credited:
                tags.add("credit_note")
            inv["amount"], inv["credited"] = amount, credited
            remaining = money(amount - credited)
            paid = remaining if mode == "full" else money(remaining * rng.choice([0.25, 0.4, 0.5, 0.6])) \
                if mode == "partial" else 0.0
            if paid:
                tags.add("fully_paid_excluded" if mode == "full" else "part_paid")
                halves = [paid] if rng.random() < 0.6 else [money(paid * 0.5), money(paid - money(paid * 0.5))]
                parts += [(inv, h) for h in halves]
        sister = [self._invoice(rng, self.sister, multipage=False) for _ in range(rng.randrange(2, 4))]
        parts += [(inv, money(inv["total"] * 0.5)) for inv in sister if rng.random() < 0.6]
        self._remittances(rng, parts)
        self._on_account(rng, [inv for i, inv in enumerate(invoices) if i not in cancelled])
        for _ in range(rng.randrange(2, 4)):
            self._invoice(rng, synth.company(rng, taken) + " " + rng.choice(SUFFIXES)[0], multipage=False)
        self.expected_invoices = []
        for i, inv in enumerate(invoices):
            if i in cancelled:
                continue
            outstanding = money(inv["amount"] - inv["credited"] - inv["paid"])
            if outstanding > 0.004:
                rate = 1.0 if inv["currency"] == "GBP" else self.rates[inv["currency"]][inv["issue"]]
                self.expected_invoices.append({"invoice_number": inv["number"], "currency": inv["currency"],
                                               "outstanding": outstanding, "outstanding_gbp": money(outstanding * rate),
                                               "bucket": bucket((self.as_of - inv["due"]).days)})
        self.expected = {"vendor": self.vendor, "as_of": self.as_of.isoformat(),
                         "invoices": sorted(self.expected_invoices, key=lambda i: i["invoice_number"]),
                         "total_outstanding_gbp": money(sum(i["outstanding_gbp"] for i in self.expected_invoices))}
        self.prompt = (
            f"{self.header()} Prepare the open-items ledger for our supplier {self.vendor} (VAT ID {self.vat}) as of "
            f"{self.as_of.isoformat()} and submit it with submit_ledger. Use search_documents and get_document. Rules: "
            f"documents of other companies, including companies with a similar name or a different VAT ID, do not "
            f"count. An invoice's amount is its final total; a later dated amendment replaces it, and the latest "
            f"amendment wins. A cancelled invoice has nothing outstanding; a duplicate copy is the same invoice, not "
            f"a new one. Outstanding = amount - credit notes - payments, in the invoice currency; a payment made in "
            f"another currency counts as the paid amount converted into the invoice currency with the rate dated "
            f"on the payment date (GBP amount / rate), rounded to 2 decimals. A payment on account (no invoice "
            f"reference) settles the open invoices in its currency in order of due date, oldest first, each in full "
            f"before the next, after every other payment and credit. List only invoices with an "
            f"outstanding balance. Convert to GBP with the rate dated on the invoice date in the "
            f"exchange-rate table, rounding to 2 decimals. Age by days past the due date as of the reference date: "
            f"current (not yet overdue), 1-30, 31-60 or 61+. total_outstanding_gbp is the sum of outstanding_gbp.")

    # ------------------------------------------------------------ documents --

    def _add(self, kind: str, title: str, date: dt.date, text: str) -> str:
        while True:
            doc_id = f"DOC-{self.rng.randrange(100000, 999999)}"
            if doc_id not in self.docs:
                break
        self.docs[doc_id] = {"type": kind, "title": title, "date": date.isoformat(), "text": text}
        return doc_id

    def _rates(self, rng) -> dict:
        start = dt.date(2026, 1, 2)
        days = [start + dt.timedelta(days=d) for d in range(0, (self.as_of - start).days)
                if (start + dt.timedelta(days=d)).weekday() < 5]
        rates = {"EUR": {}, "USD": {}}
        eur, usd = rng.uniform(0.83, 0.88), rng.uniform(0.72, 0.8)
        for d in days:
            eur *= rng.uniform(0.995, 1.005)
            usd *= rng.uniform(0.995, 1.005)
            rates["EUR"][d], rates["USD"][d] = round(eur, 4), round(usd, 4)
        self.rate_days = days
        rows = "\n".join(f" {d.isoformat()} | {rates['EUR'][d]:.4f} | {rates['USD'][d]:.4f}" for d in days)
        self._add("exchange rates", "Treasury exchange-rate table 2026 (GBP per unit)", days[-1],
                  "NORTHWIND ANALYTICS LTD — TREASURY\nDaily reference rates, GBP per one unit of currency. Use the "
                  f"rate dated on the document date.\n\n date       | EUR    | USD\n{rows}\n")
        return rates

    def _letterhead(self, company: str) -> str:
        """One fixed letterhead per company; the sister company has its own address and VAT ID."""
        if company not in self.letterheads:
            rng = self.rng
            vat = self.vat if company == self.vendor else f"EU{rng.randrange(10**8, 10**9)}"
            self.letterheads[company] = (f"{company.upper()}\n{synth.street(rng)} · {rng.randrange(10000, 99999)} "
                                         f"{synth.city(rng, set())}\nVAT ID: {vat}\n")
        return self.letterheads[company]

    def _dated(self, date: dt.date) -> dt.date:
        """Every movement happens before the reference date."""
        return min(date, self.as_of - dt.timedelta(days=1))

    def _invoice(self, rng, company: str, multipage: bool) -> dict:
        currency = rng.choice(["GBP", "EUR", "EUR", "USD"])
        issue = rng.choice(self.rate_days[:-60])
        due = issue + dt.timedelta(days=rng.choice([14, 30, 45, 60]))
        while True:
            number = f"INV-{rng.randrange(100000, 999999)}"
            if number not in self.numbers:
                self.numbers.add(number)
                break
        lines = [(f"{rng.choice(['AX', 'BK', 'CR', 'DV', 'EL', 'FM'])}-{rng.randrange(1000, 9999)}", rng.choice(PRODUCTS),
                  rng.randrange(1, 40), rng.randrange(150, 60000) / 100) for _ in range(rng.randrange(4, 9))]
        subtotal = money(sum(q * p for _, _, q, p in lines))
        tax = money(subtotal * 0.2)
        total = money(subtotal + tax)
        fmt = lambda rows: "\n".join(f" {s:<8} | {d:<22} | {q:>4} | {p:>9.2f} | {q * p:>10.2f}" for s, d, q, p in rows)  # noqa: E731
        head = (self._letterhead(company) + f"\nINVOICE {number}\nInvoice date: {issue.isoformat()}\nDue date: "
                f"{due.isoformat()}\nCurrency: {currency}\n\n SKU      | Description            |  Qty |     Price |     Amount\n")
        terms = noisy(rng, synth.boilerplate(rng, rng.randrange(150, 350)))
        totals = f"Subtotal: {subtotal:.2f}\nVAT 20%: {tax:.2f}\nTOTAL: {currency} {total:.2f}\n"
        full = head + fmt(lines) + "\n\n" + totals
        if multipage:
            cut = len(lines) // 2
            self._add("invoice", f"{company} — {number} (page 1 of 2)", issue,
                      head + fmt(lines[:cut]) + "\n\n(continued on page 2)\n\n" + terms)
            self._add("invoice", f"{company} — {number} (page 2 of 2)", issue,
                      self._letterhead(company) + f"\nINVOICE {number} — page 2 of 2\n\n" + fmt(lines[cut:]) + "\n\n"
                      + totals + "\n" + noisy(rng, synth.boilerplate(rng, rng.randrange(80, 200))))
        else:
            self._add("invoice", f"{company} — {number}", issue, head + fmt(lines) + "\n\n" + totals + "\n" + terms)
        return {"number": number, "currency": currency, "issue": issue, "due": due, "total": total, "company": company,
                "text": full, "paid": 0.0}

    def _copy(self, rng, inv: dict) -> None:
        sent = self._dated(inv["issue"] + dt.timedelta(days=rng.randrange(7, 30)))
        self._add("invoice", f"{inv['company']} — {inv['number']} (copy)", sent,
                  f"DUPLICATE COPY sent {sent.isoformat()} at the customer's request - this is not a new invoice.\n\n"
                  + inv["text"])

    def _cancel(self, rng, inv: dict) -> None:
        date = self._dated(inv["issue"] + dt.timedelta(days=rng.randrange(2, 15)))
        self._add("cancellation", f"{inv['company']} — cancellation of {inv['number']}", date,
                  self._letterhead(inv["company"]) + f"\nCANCELLATION NOTICE\nDate: {date.isoformat()}\nInvoice "
                  f"{inv['number']} dated {inv['issue'].isoformat()} is cancelled in full and must not be paid. The "
                  f"goods were never dispatched.\n\n" + noisy(rng, synth.boilerplate(rng, 80)))

    def _on_account(self, rng, invoices: list) -> None:
        """A payment on account with no invoice reference, applied oldest due date first (same currency)."""
        cur = rng.choice(["EUR", "USD", "GBP"])
        open_items = sorted((inv for inv in invoices if inv["currency"] == cur
                             and money(inv["amount"] - inv["credited"] - inv["paid"]) > 0.004),
                            key=lambda inv: (inv["due"], inv["number"]))
        if len(open_items) < 2:
            return
        first, second = open_items[0], open_items[1]
        lump = money(money(first["amount"] - first["credited"] - first["paid"])
                     + money(second["amount"] - second["credited"] - second["paid"]) * rng.choice([0.3, 0.5, 0.7]))
        left = lump
        for inv in open_items:
            if left <= 0.004:
                break
            take = min(left, money(inv["amount"] - inv["credited"] - inv["paid"]))
            inv["paid"] = money(inv["paid"] + take)
            left = money(left - take)
            self.inv_tags[inv["number"]].add("payment_on_account")
        date = self._rate_day(self.as_of - dt.timedelta(days=rng.randrange(2, 10)))
        self._add("remittance advice", f"Remittance advice — {self.vendor} — {date.isoformat()}", date,
                  f"NORTHWIND ANALYTICS LTD — REMITTANCE ADVICE\nPayee: {self.vendor}\nPayment date: {date.isoformat()}\n"
                  f"Payment on account (no invoice reference): {cur} {lump:.2f}\n\n" + noisy(rng, synth.boilerplate(rng, 60)))

    def _rate_day(self, date: dt.date) -> dt.date:
        """The latest rate-table day on or before `date`."""
        return max(d for d in self.rate_days if d <= date)

    def _amendment(self, rng, inv: dict) -> float:
        amended = money(inv["total"] * rng.choice([0.9, 0.95, 1.05, 1.1]))
        date = self._dated(inv["issue"] + dt.timedelta(days=rng.randrange(3, 20)))
        self._add("amendment", f"{inv['company']} — amendment to {inv['number']}", date,
                  self._letterhead(inv["company"]) + f"\nAMENDMENT to invoice {inv['number']} dated "
                  f"{inv['issue'].isoformat()}\nAmendment date: {date.isoformat()}\nA quantity on the original invoice "
                  f"was corrected. The amended invoice total is {inv['currency']} {amended:.2f} (replaces the original "
                  f"total {inv['total']:.2f}). Due date unchanged.\n\n" + noisy(rng, synth.boilerplate(rng, 120)))
        if rng.random() < 0.35:  # an earlier, superseded draft amendment
            draft = money(inv["total"] * rng.choice([0.8, 1.2]))
            self._add("amendment", f"{inv['company']} — amendment to {inv['number']}", date - dt.timedelta(days=2),
                      self._letterhead(inv["company"]) + f"\nAMENDMENT to invoice {inv['number']}\nAmendment date: "
                      f"{(date - dt.timedelta(days=2)).isoformat()}\nThe amended invoice total is {inv['currency']} "
                      f"{draft:.2f}.\n\n" + noisy(rng, synth.boilerplate(rng, 80)))
        return amended

    def _credit(self, rng, inv: dict, amount: float) -> float:
        credit = money(amount * rng.choice([0.05, 0.1, 0.15]))
        date = self._dated(inv["issue"] + dt.timedelta(days=rng.randrange(5, 25)))
        number = f"CN-{rng.randrange(10000, 99999)}"
        self._add("credit note", f"{inv['company']} — credit note {number}", date,
                  self._letterhead(inv["company"]) + f"\nCREDIT NOTE {number}\nDate: {date.isoformat()}\nReference: "
                  f"invoice {inv['number']}\nReason: damaged items returned.\nCredit amount: {inv['currency']} "
                  f"{credit:.2f}\n\n" + noisy(rng, synth.boilerplate(rng, 100)))
        return credit

    def _remittances(self, rng, parts: list) -> None:
        """Group payment parts into remittance advices of one to three lines per payee; a line against a EUR or
        USD invoice is sometimes paid in GBP and reduces the invoice by the GBP amount at that day's rate."""
        rng.shuffle(parts)
        by_payee: dict[str, list] = {}
        for inv, part in parts:
            by_payee.setdefault(inv["company"], []).append((inv, part))
        for payee, items in sorted(by_payee.items()):
            while items:
                n = min(len(items), rng.choice([1, 1, 2, 3]))
                group, items = items[:n], items[n:]
                if len({inv["number"] for inv, _ in group}) < n:  # one invoice per advice
                    group, items = group[:1], group[1:] + items
                date = self._rate_day(self._dated(max(inv["issue"] + dt.timedelta(days=rng.randrange(10, 50))
                                                      for inv, _ in group)))
                lines = []
                for inv, part in group:
                    tags = self.inv_tags.setdefault(inv["number"], set())
                    if n > 1:
                        tags.add("multi_invoice_remittance")
                    if inv["currency"] != "GBP" and rng.random() < 0.4:
                        tags.add("paid_in_gbp")
                        rate = self.rates[inv["currency"]][date]
                        gbp = money(part * rate)
                        inv["paid"] = money(inv["paid"] + money(gbp / rate))
                        lines.append(f"Applied to invoice {inv['number']}: GBP {gbp:.2f}")
                    else:
                        inv["paid"] = money(inv["paid"] + part)
                        lines.append(f"Applied to invoice {inv['number']}: {inv['currency']} {part:.2f}")
                self._add("remittance advice", f"Remittance advice — {payee} — {date.isoformat()}", date,
                          f"NORTHWIND ANALYTICS LTD — REMITTANCE ADVICE\nPayee: {payee}\nPayment date: "
                          f"{date.isoformat()}\n" + "\n".join(lines) + "\n\n" + noisy(rng, synth.boilerplate(rng, 60)))

    # ---------------------------------------------------------------- tools --

    def tool_search_documents(self, query: str, page: int = 1) -> str:
        terms = [t for t in re.findall(r"[a-z0-9]+(?:[-.][a-z0-9]+)*", query.lower()) if len(t) > 1]
        if not terms:
            raise ToolError("the query has no searchable terms")
        scored = []
        for doc_id, d in self.docs.items():
            words = set(re.findall(r"[a-z0-9]+(?:[-.][a-z0-9]+)*", (d["title"] + " " + d["text"]).lower()))
            score = sum(2 if t in d["title"].lower() else 1 for t in terms if t in words)
            if score:
                scored.append((-score, d["date"], doc_id))
        if not scored:
            return "no matches"
        pages = (len(scored) + 5) // 6
        if not 1 <= page <= pages:
            raise ToolError(f"page must be between 1 and {pages}")
        rows = sorted(scored)[(page - 1) * 6:page * 6]
        return "\n".join([f"page {page} of {pages} ({len(scored)} matches)"] + [
            f"{i} | {self.docs[i]['type']} | {self.docs[i]['title']} | {self.docs[i]['date']}" for _, _, i in rows])

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
        if self.submitted is None:
            self.requirements = {f"{self.kind}.submitted": False}
            return False, "no submission"
        return self._check_ledger() if self.kind == "ledger" else self._check_deal()

    def _check_deal(self) -> tuple[bool, str]:
        got = self.submitted
        self.requirements = {}
        for name, want in self.expected.items():
            spec = self.fields[name]
            self.requirements[f"{self.kind}.{name}"] = (
                rows_match(got.get(name), want, spec[2], spec[3]) if spec[0] == "table"
                else same(self.types[name], got.get(name), want))
        wrong = [k.split(".", 1)[1] for k, ok in self.requirements.items()
                 if (ok[0] != ok[1] if isinstance(ok, list) else not ok)]
        return not wrong, "all fields match" if not wrong else "wrong: " + ", ".join(wrong[:8])

    def _check_ledger(self) -> tuple[bool, str]:
        got, want = self.submitted, self.expected
        wrong = []
        norm = lambda s: " ".join(re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).split())  # noqa: E731
        req = {"ledger.vendor": norm(got.get("vendor")) == norm(want["vendor"]),
               "ledger.as_of": str(got.get("as_of", "")).strip() == want["as_of"]}
        have = {str(i.get("invoice_number", "")).strip().upper(): i for i in got.get("invoices") or []}
        need = {i["invoice_number"]: i for i in want["invoices"]}
        good = set()
        for number, n in need.items():
            h = have.get(number)
            if h is None:
                wrong.append(f"{number} missing")
                continue
            bad = [k for k in ("outstanding", "outstanding_gbp") if not isinstance(h.get(k), (int, float))
                   or abs(h[k] - n[k]) > 0.011] + [k for k in ("currency", "bucket") if h.get(k) != n[k]]
            wrong += [f"{number}.{k}" for k in bad]
            if not bad:
                good.add(number)
        extra = sorted(set(have) - set(need))
        wrong += [f"{number} extra" for number in extra]
        for number, tags in self.inv_tags.items():  # an excluded invoice is right when absent
            right = number not in have if number not in need else number in good
            for tag in tags or {"plain"}:
                row = req.setdefault(f"ledger.{tag}", [0, 0])  # [invoices right, invoices of this kind]
                row[0] += right
                row[1] += 1
        req["ledger.no_other_invoices"] = not (set(extra) - set(self.inv_tags))
        total = got.get("total_outstanding_gbp")
        req["ledger.total"] = isinstance(total, (int, float)) and abs(total - want["total_outstanding_gbp"]) <= 0.03
        wrong += [k.split(".", 1)[1] for k in ("ledger.vendor", "ledger.as_of", "ledger.total") if not req[k]]
        req["ledger.no_other_invoices"] = [int(req["ledger.no_other_invoices"]), 1]
        self.requirements = req
        return not wrong, "all fields match" if not wrong else "wrong: " + ", ".join(wrong[:8])

    def oracle(self) -> list[Step]:
        if self.kind == "ledger":
            mine = [i for i, d in self.docs.items() if self.vendor.upper() in d["text"] or self.vendor in d["text"]
                    or d["type"] == "exchange rates"]
            query = self.vendor
        else:
            mine = sorted(self.docs)
            query = self.expected["supplier_legal_name"]
        return [Step(calls=[Call("search_documents", {"query": query})]),
                Step(calls=[Call("get_document", {"doc_id": i}) for i in sorted(mine)]),
                Step(calls=[Call(self.submit_tool, self.expected)]),
                Step(content="Submitted.")]


def norm_name(value) -> str:
    text = str(value).lower().replace("&", " and ")
    text = " ".join(re.sub(r"[^a-z0-9 ]", " ", text).split())
    return text[4:] if text.startswith("the ") else text


def same(kind: str, got, want) -> bool:
    """One deal field: dates and codes exactly, amounts to the cent, names normalized, lists as sets."""
    if got is None:
        return False
    if kind in ("money", "number"):
        return isinstance(got, (int, float)) and not isinstance(got, bool) and abs(got - want) <= 0.011
    if kind == "int":
        return isinstance(got, int) and not isinstance(got, bool) and got == want
    if kind == "bool":
        return got is want
    if kind == "date":
        return str(got).strip() == want
    if kind == "code":
        return re.sub(r"\s", "", str(got)).upper() == re.sub(r"\s", "", str(want)).upper()
    if kind == "name":
        return norm_name(got) == norm_name(want)
    if kind == "names":
        return isinstance(got, list) and sorted(map(norm_name, got)) == sorted(map(norm_name, want))
    raise ValueError(kind)
