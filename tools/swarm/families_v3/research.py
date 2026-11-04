"""Research (hard tier): Byline-style research briefs over a ~210-page generated encyclopedia.

The world has 24 companies with chief-executive histories, headquarters moves,
acquisitions (some by subsidiaries, three levels deep), renamed companies (listed
under the old name in earlier events), near-duplicate company names, archived page
revisions that disagree with the current page, product version histories and
revenue tables. Search returns at most 5 snippets. The agent submits a structured
brief - a company profile, a person profile, a market brief for a country, or a
group (M&A) brief - and cites the page each single-source fact comes from. Every
field and every citation is one requirement.
"""

from __future__ import annotations

import re

from families import Call, Step, Task, ToolError, rng_for, synth, tool

from .rows import norm, rows_match

YEARS = list(range(2018, 2026))  # revenue table years
CATEGORIES = ["industrial sensor", "handheld scanner", "battery pack", "routing appliance", "lab centrifuge",
              "drone controller", "smart thermostat", "label printer", "water filter", "solar inverter"]
VERSION_NAMES = ["", "Mk II", "Mk III", "Pro"]
FILLER = ["This {kind} is described in several trade publications.",
          "A 2019 profile of {name} focused on its suppliers in {place}.",
          "The first public mention of {name} appears in a regional newspaper.",
          "Historians disagree about how much {name} shaped the economy of {place}.",
          "Interviews about {name} were collected for an oral history project in {place}.",
          "Trade statistics from {place} sometimes list {name} under a different category.",
          "An annual award in {place} once carried the name of {name}.",
          "Analysts have compared {name} with regional peers on several occasions.",
          "Records about {name} were digitised as part of a public archive project.",
          "Retrospectives on {name} usually emphasise continuity rather than change.",
          "A museum exhibit in {place} once featured material related to {name}.",
          "Several university case studies discuss {name}."]


def slug(text: str) -> str:
    return "wiki/" + re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")



CURRENT = 2025  # "latest" revenue year; roles still open are counted to 2026


def table(desc: str, key: str, **columns: str) -> tuple:
    """A table field: rows of named columns (string or integer), matched by their `key` column."""
    return ("table", desc, key, columns)


def brief_tool(name: str, title: str, fields: dict, sourced: list[str]) -> dict:
    props = {}
    for key, spec in fields.items():
        kind, desc = spec[0], spec[1]
        if kind == "table":
            columns = spec[3]
            props[key] = {"type": "array", "description": desc, "items": {
                "type": "object", "properties": {c: {"type": t} for c, t in columns.items()},
                "required": list(columns), "additionalProperties": False}}
        elif kind == "array":
            props[key] = {"type": "array", "items": {"type": "string"}, "description": desc}
        else:
            props[key] = {"type": kind, "description": desc}
    props["sources"] = {"type": "object", "description": "the page id each of these facts comes from",
                        "properties": {k: {"type": "string"} for k in sourced}, "required": sourced,
                        "additionalProperties": False}
    return tool(name, f"Submit the {title} (ends the task).", props, list(fields) + ["sources"])


COMPANY = {"current_name": ("string", "the company's current name"),
           "former_name": ("string", "its previous name, or none"),
           "founded": ("integer", "year founded"),
           "headquarters_city": ("string", "current headquarters city"),
           "headquarters_country": ("string", "country of the current headquarters"),
           "chief_executive": ("string", "current chief executive"),
           "chief_executive_since": ("integer", "year the current chief executive took over"),
           "parent_company": ("string", "the company that owns it directly (current name), or none"),
           "revenue_2025": ("integer", "millions of dollars"),
           "revenue_growth_pct": ("number", "growth from 2022 to 2025 in percent, 1 decimal"),
           "previous_chief_executive": ("string", "the chief executive before the current one, or none"),
           "headquarters_moved_from": ("string", "city of the previous headquarters"),
           "headquarters_move_year": ("integer", "year it moved"),
           "acquisitions": ("array", "companies it acquired itself (current names), empty if none"),
           "products": ("array", "its product lines"),
           "latest_launch": ("string", "the product version launched most recently"),
           "latest_launch_price": ("integer", "that version's list price in dollars"),
           "ceo_history": table("every chief executive since 2005, oldest first; to is a year or \"present\"", "name",
                                name="string", **{"from": "integer"}, to="string")}
PERSON = {"name": ("string", "the person's name"),
          "born": ("integer", "birth year"),
          "birthplace": ("string", "birth city"),
          "birthplace_country": ("string", "country of the birth city"),
          "lives_in": ("string", "city of residence"),
          "current_company": ("string", "company they lead today (current name), or none"),
          "companies_led": ("array", "every company they have led as chief executive (current names)"),
          "first_year_as_ceo": ("integer", "year of their first chief-executive role"),
          "years_as_ceo": ("integer", "total years as chief executive, counting open roles to 2026"),
          "current_company_country": ("string", "country of their current company's headquarters, or none"),
          "career": table("every chief-executive role, oldest first (current company names); to is a year or "
                          "\"present\"", "company", company="string", **{"from": "integer"}, to="string")}
MARKET = {"country": ("string", "the country"),
          "companies": ("array", "every company headquartered there today (current names)"),
          "company_count": ("integer", "how many"),
          "largest_by_revenue_2025": ("string", "the one with the highest 2025 revenue"),
          "largest_revenue_2025": ("integer", "its 2025 revenue, millions of dollars"),
          "combined_revenue_2025": ("integer", "their combined 2025 revenue, millions of dollars"),
          "oldest_company": ("string", "the one founded earliest"),
          "oldest_founded": ("integer", "its founding year"),
          "fastest_growing": ("string", "the one with the highest revenue growth from 2022 to 2025"),
          "moved_in": ("array", "those whose previous headquarters was in another country, empty if none"),
          "company_table": table("one row per company headquartered there", "company", company="string",
                                 chief_executive="string", revenue_2025="integer")}
GROUP = {"owner": ("string", "the group's parent company"),
         "direct_subsidiaries": ("array", "companies it acquired itself (current names)"),
         "indirect_subsidiaries": ("array", "companies acquired by its subsidiaries, at any depth (current names)"),
         "most_recent_acquisition": ("string", "the group's latest acquisition (current name)"),
         "most_recent_acquisition_year": ("integer", "its year"),
         "group_revenue_2025": ("integer", "owner plus every subsidiary, millions of dollars"),
         "largest_subsidiary_2025": ("string", "the subsidiary with the highest 2025 revenue"),
         "renamed_subsidiaries": ("array", "subsidiaries that changed their name (current names), empty if none"),
         "subsidiary_countries": ("array", "countries where the subsidiaries are headquartered today"),
         "subsidiary_table": table("one row per subsidiary, direct or indirect", "company", company="string",
                                   parent="string", acquired_year="integer", revenue_2025="integer")}
BRIEFS = {"company": (COMPANY, ["current_name", "former_name", "founded", "headquarters_city", "headquarters_country",
                                "chief_executive", "chief_executive_since", "parent_company", "revenue_2025",
                                "revenue_growth_pct", "previous_chief_executive", "headquarters_moved_from",
                                "acquisitions", "products", "latest_launch", "latest_launch_price"],
                      "submit_company_brief",
                      "company profile brief"),
          "person": (PERSON, ["born", "birthplace", "birthplace_country", "lives_in", "current_company",
                              "companies_led", "first_year_as_ceo", "years_as_ceo", "current_company_country"],
                     "submit_person_brief",
                     "person profile brief"),
          "market": (MARKET, ["largest_by_revenue_2025", "largest_revenue_2025", "oldest_company", "oldest_founded",
                              "fastest_growing"],
                     "submit_market_brief", "market brief"),
          "group": (GROUP, ["direct_subsidiaries", "most_recent_acquisition", "most_recent_acquisition_year",
                            "largest_subsidiary_2025"], "submit_group_brief", "group brief")}
ASK = {"company": "a company profile of {subject}",
       "person": "a profile of {subject}",
       "market": "a market brief on the companies headquartered in {subject} today",
       "group": "a group brief on {subject} and every company it owns directly or through its subsidiaries"}


class Research3Task(Task):
    family = "research3"
    max_turns = 40
    submit_tool = "submit_company_brief"
    tools = [
        tool("search", "Search the encyclopedia, best matches first, 5 per page, as page id | title | snippet.",
             {"query": {"type": "string"}, "page": {"type": "integer", "description": "1-based, default 1"}},
             ["query"]),
        tool("open_page", "Return the full text of an encyclopedia page.", {"page_id": {"type": "string"}}, ["page_id"]),
        *(brief_tool(name, title, fields, sourced) for fields, sourced, name, title in BRIEFS.values()),
    ]
    latency = {"search": 1.5, "open_page": 1.5}
    VARIANTS = ["company", "person", "market", "group"]

    def __init__(self, index: int, seed: int):
        super().__init__(index, seed)
        self.kind = self.VARIANTS[index % 4]
        self.tier = "hard"
        for attempt in range(80):  # regenerate the world until the brief is well-posed
            rng = rng_for(self.family, seed, index, attempt)
            self._world(rng)
            brief = getattr(self, "_" + self.kind)(rng)
            if brief:
                break
        else:
            raise RuntimeError(f"{self.id}: no well-posed brief")
        self.subject, self.expected, self.support = brief
        self._pages(rng)
        fields, sourced, self.submit_tool, title = BRIEFS[self.kind]
        self.fields, self.sourced = fields, sourced
        self.prompt = (
            f"{self.header()} Research brief: prepare {ASK[self.kind].format(subject=self.subject)} and submit it with "
            f"{self.submit_tool}. Use search and open_page; rely only on the encyclopedia. Current pages are "
            f"authoritative; archived revisions describe the past. Companies are named by their current names. For "
            f"each fact listed under sources, give the id of the page that states it (a current page, not an archived "
            f"revision). Revenue figures are in millions of dollars; revenue_growth_pct is (2025 - 2022) / 2022 x "
            f"100, rounded to 1 decimal. Use none where a company has no former name or no parent.")

    # ------------------------------------------------------------ briefs --

    def owner_of(self, company: str) -> str | None:
        return next((o for o, c in self.companies.items() if any(t == company for t, _ in c["acquired"])), None)

    def _company(self, rng):
        page, fin = slug, (lambda n: slug(f"Financial summary {n}"))
        names = [n for n, c in sorted(self.companies.items())
                 if sum([bool(c["former_name"]), bool(self.owner_of(n)), c["hq_moved"] > 2012]) >= 2]
        if not names:
            return None
        name = rng.choice(names)
        c = self.companies[name]
        ceo, since, _ = c["ceos"][-1]
        launches = sorted(((v[1], v[0], p) for p in c["products"] for v in self.products[p]["versions"]), reverse=True)
        if len(launches) > 1 and launches[0][0] == launches[1][0]:
            return None
        owner = self.owner_of(name)
        r22, r25 = c["revenue"][2022], c["revenue"][CURRENT]
        previous = c["ceos"][-2][0] if len(c["ceos"]) > 1 else None
        launch_price = next(v[2] for v in self.products[launches[0][2]]["versions"] if v[0] == launches[0][1])
        expected = {"current_name": name, "former_name": c["former_name"] or "none", "founded": c["founded"],
                    "ceo_history": [{"name": n, "from": a, "to": str(b) if b else "present"} for n, a, b in c["ceos"]],
                    "previous_chief_executive": previous or "none", "headquarters_moved_from": c["hq_old"],
                    "headquarters_move_year": c["hq_moved"], "acquisitions": sorted(t for t, _ in c["acquired"]),
                    "latest_launch_price": launch_price,
                    "headquarters_city": c["hq"], "headquarters_country": self.cities[c["hq"]]["country"],
                    "chief_executive": ceo, "chief_executive_since": since, "parent_company": owner or "none",
                    "revenue_2025": r25, "revenue_growth_pct": round((r25 - r22) / r22 * 100 + 1e-9, 1),
                    "products": sorted(c["products"]), "latest_launch": launches[0][1]}
        support = {"current_name": {page(name)}, "former_name": {page(name)} if c["former_name"] else set(),
                   "founded": {page(name)},
                   "headquarters_city": {page(name)}, "headquarters_country": {page(c["hq"])},
                   "chief_executive": {page(name), page(ceo)}, "chief_executive_since": {page(name), page(ceo)},
                   "parent_company": {page(owner)} if owner else set(), "revenue_2025": {fin(name)},
                   "revenue_growth_pct": {fin(name)}, "products": {page(name)}, "latest_launch": {page(launches[0][2])},
                   "previous_chief_executive": {page(name)} if previous else set(), "headquarters_moved_from": {page(name)},
                   "acquisitions": {page(name)} if c["acquired"] else set(), "latest_launch_price": {page(launches[0][2])}}
        return name, expected, support

    def _person(self, rng):
        people = [n for n, p in sorted(self.people.items()) if p["roles"] and (
            len(p["roles"]) > 1 or any(self.companies[c]["former_name"] for c, _, _ in p["roles"]))]
        if not people:
            return None
        name = rng.choice(people)
        p = self.people[name]
        current = next((c for c, _, e in p["roles"] if e is None), None)
        expected = {"name": name, "born": p["born"], "birthplace": p["birthplace"],
                    "birthplace_country": self.cities[p["birthplace"]]["country"], "lives_in": p["lives"],
                    "current_company": current or "none", "companies_led": sorted({c for c, _, _ in p["roles"]}),
                    "first_year_as_ceo": min(s for _, s, _ in p["roles"]),
                    "years_as_ceo": sum((e or 2026) - s for _, s, e in p["roles"]),
                    "current_company_country": self.cities[self.companies[current]["hq"]]["country"] if current else "none",
                    "career": [{"company": c, "from": a, "to": str(b) if b else "present"}
                               for c, a, b in sorted(p["roles"], key=lambda r: r[1])]}
        support = {k: {slug(name)} for k in expected}
        support["birthplace_country"] = {slug(p["birthplace"])}
        support["current_company"] = {slug(name), slug(current)} if current else set()  # nothing to cite for none
        support["current_company_country"] = {slug(self.companies[current]["hq"])} if current else set()
        listed = [c for c, s, e in p["roles"] if (name, s, e) in self.companies[c]["ceos"]]
        if len(p["roles"]) == 1 and listed:  # a single role is stated on the company's page too
            for key in ("companies_led", "first_year_as_ceo", "years_as_ceo"):
                support[key].add(slug(listed[0]))
        first = min(p["roles"], key=lambda r: r[1])
        if first[0] in listed:
            support["first_year_as_ceo"].add(slug(first[0]))
        return name, expected, support

    def _market(self, rng):
        country = rng.choice(sorted({c["country"] for c in self.cities.values()}))
        here = sorted(n for n, c in self.companies.items() if self.cities[c["hq"]]["country"] == country)
        if len(here) < 4:
            return None
        by_rev = sorted(here, key=lambda n: -self.companies[n]["revenue"][CURRENT])
        by_age = sorted(here, key=lambda n: self.companies[n]["founded"])
        if (self.companies[by_rev[0]]["revenue"][CURRENT] == self.companies[by_rev[1]]["revenue"][CURRENT]
                or self.companies[by_age[0]]["founded"] == self.companies[by_age[1]]["founded"]):
            return None
        if not any(self.companies[n]["hq_moved"] > 2012 for n in here):
            return None
        growth = lambda n: (self.companies[n]["revenue"][CURRENT] - self.companies[n]["revenue"][2022]) / \
            self.companies[n]["revenue"][2022]  # noqa: E731
        by_growth = sorted(here, key=lambda n: -growth(n))
        if abs(growth(by_growth[0]) - growth(by_growth[1])) < 0.005:
            return None
        expected = {"country": country, "companies": here, "company_count": len(here),
                    "largest_by_revenue_2025": by_rev[0],
                    "largest_revenue_2025": self.companies[by_rev[0]]["revenue"][CURRENT],
                    "combined_revenue_2025": sum(self.companies[n]["revenue"][CURRENT] for n in here),
                    "oldest_company": by_age[0], "oldest_founded": self.companies[by_age[0]]["founded"],
                    "fastest_growing": by_growth[0],
                    "company_table": [{"company": n, "chief_executive": self.companies[n]["ceos"][-1][0],
                                       "revenue_2025": self.companies[n]["revenue"][CURRENT]} for n in here],
                    "moved_in": sorted(n for n in here if self.cities[self.companies[n]["hq_old"]]["country"] != country)}
        support = {"largest_by_revenue_2025": {slug(f"Financial summary {by_rev[0]}")},
                   "largest_revenue_2025": {slug(f"Financial summary {by_rev[0]}")},
                   "oldest_company": {slug(by_age[0])}, "oldest_founded": {slug(by_age[0])},
                   "fastest_growing": {slug(f"Financial summary {by_growth[0]}")}}
        return country, expected, support

    def _group(self, rng):
        owners = [n for n in sorted(self.companies) if len(self.group(n, 2100)) >= 3 and not self.owner_of(n)]
        if not owners:
            return None
        owner = rng.choice(owners)
        members = self.group(owner, 2100)
        direct = sorted(t for t, _ in self.companies[owner]["acquired"])
        deals = sorted(((y, t, o) for o in [owner] + members for t, y in self.companies[o]["acquired"]), reverse=True)
        if len(deals) > 1 and deals[0][0] == deals[1][0]:
            return None
        biggest = sorted(members, key=lambda m: -self.companies[m]["revenue"][CURRENT])
        if self.companies[biggest[0]]["revenue"][CURRENT] == self.companies[biggest[1]]["revenue"][CURRENT]:
            return None
        year, latest, buyer = deals[0]
        expected = {"owner": owner, "direct_subsidiaries": direct,
                    "indirect_subsidiaries": sorted(m for m in members if m not in direct),
                    "most_recent_acquisition": latest, "most_recent_acquisition_year": year,
                    "group_revenue_2025": sum(self.companies[m]["revenue"][CURRENT] for m in [owner] + members),
                    "largest_subsidiary_2025": biggest[0],
                    "subsidiary_table": [{"company": m, "parent": self.owner_of(m),
                                          "acquired_year": next(y for t, y in self.companies[self.owner_of(m)]["acquired"]
                                                                if t == m),
                                          "revenue_2025": self.companies[m]["revenue"][CURRENT]} for m in members],
                    "renamed_subsidiaries": sorted(m for m in members if self.companies[m]["former_name"]),
                    "subsidiary_countries": sorted({self.cities[self.companies[m]["hq"]]["country"] for m in members})}
        support = {"direct_subsidiaries": {slug(owner)}, "most_recent_acquisition": {slug(buyer)},
                   "most_recent_acquisition_year": {slug(buyer)},
                   "largest_subsidiary_2025": {slug(f"Financial summary {biggest[0]}")}}
        return owner, expected, support

    def _world(self, rng) -> None:
        taken: set = set()
        countries = [synth.word(rng, 2) + rng.choice(["ia", "land", "ovia", "ar"]) for _ in range(3)]
        self.cities = {}
        for i in range(15):
            self.cities[synth.city(rng, taken)] = {"country": countries[i % 3],
                                                   "population": rng.randrange(40, 2400) * 1000 + rng.randrange(1000)}
        cities = list(self.cities)
        births = [rng.randrange(1940, 2000) for _ in range(80)]
        self.people = {}
        for i in range(80):
            self.people[synth.person(rng, taken)] = {"born": births[i], "birthplace": rng.choice(cities),
                                                     "lives": rng.choice(cities), "roles": []}
        people = list(self.people)
        # 24 companies; two share a first word (near-duplicate names); four were renamed
        names = [synth.company(rng, taken) for _ in range(23)]
        twin = names[0].split()[0] + " " + rng.choice(["Dynamic Systems", "Holdings", "Technologies"])
        names.append(twin)
        rng.shuffle(names)
        self.companies = {}
        free = people[:]
        rng.shuffle(free)
        for name in names:
            hq_old, hq = rng.sample(cities, 2)
            company = {"founded": rng.randrange(1890, 2000), "hq": hq, "hq_old": hq_old, "hq_moved": rng.randrange(2012, 2024),
                       "ceos": [], "acquired": [], "products": [], "former_name": None, "renamed": None,
                       "revenue": {y: rng.randrange(20, 4000) for y in YEARS}}
            # 2-3 chief executives covering 2005 to today
            cuts = sorted(rng.sample(range(2007, 2024), rng.randrange(1, 3)))
            spans = list(zip([2005] + cuts, cuts + [None]))
            for start, end in spans:
                person = free.pop()
                company["ceos"].append((person, start, end))
                self.people[person]["roles"].append((name, start, end))
            self.companies[name] = company
        # one person later led a second company (careers cross companies)
        for name in rng.sample(names, 2):
            c = self.companies[name]
            person, start, end = c["ceos"][-1]
            if end is None and start > 2012:
                previous = rng.choice([n for n in names if n != name])
                pc = self.companies[previous]
                if pc["ceos"][0][1] == 2005 and pc["ceos"][0][2] and pc["ceos"][0][2] <= start - 2:
                    # the person had led `previous` before; shift the record only in the career list
                    self.people[person]["roles"].insert(0, (previous, 2001, 2005))
        for name in rng.sample([n for n in names if n != twin], 4):
            old = synth.company(rng, taken)
            self.companies[name].update(former_name=old, renamed=rng.randrange(2012, 2024))
        # acquisitions: some companies are owned by others, a few through a subsidiary
        owners = rng.sample(names, 7)
        targets = [n for n in names if n not in owners]
        rng.shuffle(targets)
        for i, target in enumerate(targets[:13]):
            owner = owners[i % len(owners)]
            self.companies[owner]["acquired"].append((target, rng.randrange(2008, 2026)))
        for k, owner in enumerate(owners[:3]):  # nested: a subsidiary that buys another company
            sub = self.companies[owner]["acquired"][0][0]
            self.companies[sub]["acquired"].append((targets[13 + k], rng.randrange(2010, 2026)))
        deep = self.companies[owners[0]]["acquired"][0][0]  # three levels: the subsidiary's own subsidiary buys
        self.companies[targets[13]]["acquired"].append((targets[16], rng.randrange(2012, 2026)))
        assert any(t == targets[13] for t, _ in self.companies[deep]["acquired"])
        self.products = {}
        for name, c in self.companies.items():
            for _ in range(rng.randrange(2, 4)):
                pname = synth.unique(rng, lambda: f"{synth.word(rng, 2)} {rng.randrange(2, 900)}", taken)
                years = sorted(rng.sample(range(2004, 2026), rng.randrange(2, 4)))
                versions = [(f"{pname} {VERSION_NAMES[i]}".strip(), y, rng.randrange(40, 4000)) for i, y in enumerate(years)]
                self.products[pname] = {"maker": name, "category": rng.choice(CATEGORIES), "versions": versions}
                c["products"].append(pname)

    def ceo_in(self, company: str, year: int) -> str | None:
        for person, start, end in self.companies[company]["ceos"]:
            if start <= year and (end is None or year < end):
                return person
        return None

    def group(self, company: str, before: int) -> list[str]:
        out, stack = [], [company]
        while stack:
            for target, year in self.companies[stack.pop()]["acquired"]:
                if year < before and target not in out:
                    out.append(target)
                    stack.append(target)
        return out

    def hq_in(self, company: str, year: int) -> str:
        c = self.companies[company]
        return c["hq"] if year >= c["hq_moved"] else c["hq_old"]

    def _filler(self, rng, name: str, kind: str, facts: list[str]) -> str:
        sentences = facts[:]
        for _ in range(rng.randrange(12, 26)):
            sentences.insert(rng.randrange(min(1, len(sentences)), len(sentences) + 1),
                             rng.choice(FILLER).format(name=name, kind=kind, place=rng.choice(list(self.cities))))
        return " ".join(sentences)

    def _pages(self, rng) -> None:
        self.pages = {}

        def add(title: str, text: str) -> None:
            self.pages[slug(title)] = {"title": title, "text": f"{title}\n\n{text}"}
        def called(company: str, year: int) -> str:  # past events use the name the company had then
            c = self.companies[company]
            return c["former_name"] if c["former_name"] and year < c["renamed"] else company
        for name, c in self.companies.items():
            history = "; ".join(f"{p} ({s}-{e if e else 'present'})" for p, s, e in c["ceos"])
            facts = [f"{name} is a company founded in {c['founded']}, currently headquartered in {c['hq']}.",
                     f"It moved its headquarters from {c['hq_old']} to {c['hq']} in {c['hq_moved']}.",
                     f"Chief executives: {history}."]
            if c["former_name"]:
                facts.insert(1, f"It was known as {c['former_name']} until {c['renamed']}.")
            if c["acquired"]:
                facts.append("Acquisitions: " + "; ".join(f"{called(t, y)} ({y})"
                                                          for t, y in sorted(c["acquired"], key=lambda x: x[1])) + ".")
            facts.append("Products: " + ", ".join(sorted(c["products"])) + ".")
            add(name, self._filler(rng, name, "company", facts))
            if rng.random() < 0.45:  # an archived revision from before the move
                year = rng.randrange(2010, c["hq_moved"])
                old_ceo = self.ceo_in(name, year) or c["ceos"][0][0]
                add(f"{name} (archived revision, {year})", self._filler(rng, name, "company", [
                    f"This is an archived revision of the page as it stood in {year}.",
                    f"{c['former_name'] if c['former_name'] and year < c['renamed'] else name} is headquartered in "
                    f"{c['hq_old']}.", f"Its chief executive is {old_ceo}."]))
            add(f"Financial summary {name}", f"Revenue of {name} by year, in millions of dollars:\n" + "\n".join(
                f"- {y}: {c['revenue'][y]}" for y in YEARS) + "\n\n" + self._filler(rng, name, "company", []))
        for name, p in self.people.items():
            roles = "; ".join(f"chief executive of {called(comp, e or 2026)} ({s}-{e if e else 'present'})"
                              for comp, s, e in p["roles"])
            facts = [f"{name} (born {p['born']} in {p['birthplace']}) lives in {p['lives']}."]
            facts.append(f"Career: {roles}." if roles else f"{name} works as an independent consultant.")
            add(name, self._filler(rng, name, "person", facts))
        for name, p in self.products.items():
            rows = "\n".join(f"- {v[0]}: launched {v[1]}, list price ${v[2]}" for v in p["versions"])
            add(name, f"The {name} is a {p['category']} made by {p['maker']}.\nVersions:\n{rows}\n\n"
                + self._filler(rng, name, "product", []))
        for name, c in self.cities.items():
            add(name, self._filler(rng, name, "city", [f"{name} is a city in {c['country']} with a population of "
                                                      f"{c['population']:,}."]))
        add("List of companies", "\n".join(f"- {n}" for n in sorted(self.companies)))

    def tool_search(self, query: str, page: int = 1) -> str:
        terms = [t for t in norm(query).split() if len(t) > 2]
        if not terms:
            raise ToolError("the query has no searchable terms")
        scored = []
        for page_id, found in self.pages.items():
            title, words = norm(found["title"]).split(), set(norm(found["text"]).split())
            score = sum(3 if t in title else 1 for t in terms if t in words)
            if score:
                scored.append((-score, page_id))
        if not scored:
            return "no matches"
        pages = (len(scored) + 4) // 5
        if not 1 <= page <= pages:
            raise ToolError(f"page must be between 1 and {pages}")
        out = [f"page {page} of {pages} ({len(scored)} matches)"]
        for _, page_id in sorted(scored)[(page - 1) * 5:page * 5]:
            found = self.pages[page_id]
            sentences = re.split(r"(?<=[.!?]) |\n", found["text"].split("\n\n", 1)[1])
            snippet = next((x for x in sentences if any(t in norm(x).split() for t in terms)), sentences[0])
            out.append(f"{page_id} | {found['title']} | {snippet[:200]}")
        return "\n".join(out)

    def tool_open_page(self, page_id: str) -> str:
        page = self.pages.get(page_id.strip()) or self.pages.get(slug(page_id))
        if page is None:
            raise ToolError(f"no page {page_id!r}; use a page id returned by search")
        return page["text"]

    # ----------------------------------------------------------- checker --

    def check(self, final: str) -> tuple[bool, str]:
        if self.submitted is None:
            self.requirements = {f"{self.kind}.submitted": False}
            return False, "no submission"
        got = self.submitted
        req = {}
        for name, spec in self.fields.items():
            kind = spec[0]
            have, want = got.get(name), self.expected[name]
            if kind == "table":
                req[f"{self.kind}.{name}"] = rows_match(have, want, spec[2], spec[3])
                continue
            if kind == "array":
                ok = isinstance(have, list) and sorted(norm(x) for x in have) == sorted(norm(x) for x in want)
            elif kind == "integer":
                ok = isinstance(have, int) and not isinstance(have, bool) and have == want
            elif kind == "number":
                ok = isinstance(have, (int, float)) and not isinstance(have, bool) and abs(have - want) <= 0.051
            else:
                ok = norm(have or "") == norm(want)
            req[f"{self.kind}.{name}"] = ok
        sources = got.get("sources") or {}
        for name in self.sourced:
            if not self.support[name]:  # nothing to cite (no parent, no former name)
                continue
            cited = str(sources.get(name, "")).strip()
            page = cited if cited in self.pages else slug(cited)
            req[f"{self.kind}.{name}.source"] = page in self.support[name]
        self.requirements = req
        wrong = [k.split(".", 1)[1] for k, ok in req.items()
                 if (ok[0] != ok[1] if isinstance(ok, list) else not ok)]
        return not wrong, "all fields and sources match" if not wrong else "wrong: " + ", ".join(wrong[:8])

    def call(self, name: str, arguments: dict) -> str:
        if name.startswith("submit_") and name != self.submit_tool:
            raise ToolError(f"this brief must be submitted with {self.submit_tool}")
        return super().call(name, arguments)

    def oracle(self) -> list[Step]:
        pages = sorted(set().union(*self.support.values()) | {slug("List of companies")})
        answer = dict(self.expected)
        answer["sources"] = {k: sorted(self.support[k])[0] if self.support[k] else "none" for k in self.sourced}
        return [Step(calls=[Call("search", {"query": self.subject})]),
                Step(calls=[Call("open_page", {"page_id": p}) for p in pages]),
                Step(calls=[Call(self.submit_tool, answer)]),
                Step(content="Submitted.")]
