"""Research agent: multi-hop questions over a generated encyclopedia.

Each task has its own world of invented companies, people, products and cities;
pages state facts in prose next to distractors (former executives and dated
press releases, birthplaces, announcement years, metropolitan populations).
Tiers: two hops (easy); three hops or an argmax over a company's products
(medium); an aggregate over several companies, or who led a company when one
of its products launched (hard). The answer is submitted with submit_answer.
"""

from __future__ import annotations

import re

from . import Call, Step, Task, ToolError, rng_for, synth, tool

CATEGORIES = ["industrial sensor", "handheld scanner", "battery pack", "routing appliance", "lab centrifuge",
              "drone controller", "smart thermostat", "label printer", "water filter", "solar inverter"]
FILLER = ["This {kind} is described in several trade publications.",
          "A 2019 profile of {name} focused on its relationship with suppliers in {place}.",
          "The first public mention of {name} appears in a regional newspaper.",
          "{name} has been the subject of at least one documentary.",
          "Historians disagree about how much {name} shaped the region's economy.",
          "A street in {place} was informally named after {name} for several years.",
          "Interviews about {name} were collected for an oral history project in {place}.",
          "Trade statistics from {place} sometimes list {name} under a different category.",
          "An annual award in {place} once carried the name of {name}.",
          "Retrospectives on {name} usually emphasise continuity rather than change.",
          "Local archives hold correspondence about {name} dating back several decades.",
          "Coverage of {name} increased after a widely reported trade fair appearance.",
          "Analysts have compared {name} with regional peers on several occasions.",
          "A museum exhibit in {place} once featured material related to {name}.",
          "Records about {name} were digitised as part of a public archive project.",
          "Several university case studies discuss {name}.",
          "{name} appears in a number of industry directories under slightly different spellings.",
          "Photographs of {name} from the early years are held in a private collection.",
          "Commentators noted the influence of {name} on later developments in {place}."]


def article(word: str) -> str:
    return "an" if word[0].lower() in "aeiou" else "a"


def slug(text: str) -> str:
    return "wiki/" + re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", str(text).lower()).split())


class ResearchTask(Task):
    family = "research"
    max_turns = 24
    submit_tool = "submit_answer"
    tools = [
        tool("search", "Search the encyclopedia. Returns up to 8 hits as page id | title | snippet.",
             {"query": {"type": "string"}}, ["query"]),
        tool("open_page", "Return the full text of an encyclopedia page.", {"page_id": {"type": "string"}}, ["page_id"]),
        tool("submit_answer", "Submit the final answer (ends the task). Give only the answer: a name, a year or a number.",
             {"answer": {"type": "string"}}, ["answer"]),
    ]
    latency = {"search": 1.5, "open_page": 1.5}

    def __init__(self, index: int, seed: int):
        super().__init__(index, seed)
        self.rng = rng = rng_for(self.family, seed, index)
        self._world(rng)
        kind = ["two_hop", ["three_hop", "argmax"][index // 4 % 2], "aggregate", "tenure"][index % 4]
        self.tier = {"two_hop": "easy", "three_hop": "medium", "argmax": "medium"}.get(kind, "hard")
        self.kind = kind
        question, self.answer, self.path = getattr(self, "_q_" + kind)(rng)
        self.prompt = (f"{self.header()} Research question: {question} Use search and open_page to find the facts in "
                       f"the encyclopedia (do not rely on prior knowledge), then call submit_answer with only the answer.")

    # ---------------------------------------------------------------- world --

    def _world(self, rng) -> None:
        taken: set = set()
        countries = [synth.word(rng, 2) + rng.choice(["ia", "land", "ovia", "ar"]) for _ in range(3)]
        self.cities = {}
        for i in range(10):
            name = synth.city(rng, taken)
            self.cities[name] = {"country": countries[i % 3], "population": rng.randrange(40, 2400) * 1000 + rng.randrange(1000),
                                 "founded": rng.randrange(900, 1900)}
        city_names = list(self.cities)
        self.people = {}
        births = rng.sample(range(1948, 1992), 24)
        for i in range(24):
            name = synth.person(rng, taken)
            self.people[name] = {"born": births[i], "lives": rng.choice(city_names), "birthplace": rng.choice(city_names)}
        people = list(self.people)
        self.companies = {}
        n = 9
        hqs = rng.sample(city_names, n)
        for i in range(n):
            name = synth.company(rng, taken)
            since = rng.randrange(2010, 2022)
            self.companies[name] = {"founded": rng.randrange(1890, 1990), "hq": hqs[i], "ceo": people[i],
                                    "former": people[n + i], "since": since, "from": since - rng.randrange(5, 12),
                                    "office": rng.choice([c for c in city_names if c != hqs[i]]),
                                    "industry": rng.choice(synth.INDUSTRY).lower(), "products": []}
            self.people[people[i]]["ceo_of"] = name
            self.people[people[n + i]]["former_ceo_of"] = name
        self.products = {}
        for company, info in self.companies.items():
            years = rng.sample(range(info["from"], 2026), rng.randrange(2, 5))
            for year in years:
                name = unique_product(rng, taken)
                self.products[name] = {"maker": company, "launched": year, "price": rng.randrange(40, 4000),
                                       "category": rng.choice(CATEGORIES)}
                info["products"].append(name)
        self.pages = {}
        for name, c in self.companies.items():
            self._page(name, "company", rng, [
                f"{name} is {article(c['industry'])} {c['industry']} company founded in {c['founded']} and headquartered in {c['hq']}.",
                f"Its chief executive is {c['ceo']}, in the role since {c['since']}; {c['former']} led the company "
                f"from {c['from']} until {c['since']}.",
                f"The company also operates a research office in {c['office']}.",
                "Its products include " + ", ".join(sorted(c["products"])) + "."])
        for name, p in self.people.items():
            facts = [f"{name} (born {p['born']}) was born in {p['birthplace']} and lives in {p['lives']}."]
            if "ceo_of" in p:
                facts.append(f"{name} is the chief executive of {p['ceo_of']}.")
            if "former_ceo_of" in p:
                facts.append(f"{name} formerly led {p['former_ceo_of']} and now advises several start-ups.")
                c = self.companies[p["former_ceo_of"]]
                self.pages[slug(f"Press release {p['former_ceo_of']} {c['from']}")] = {
                    "title": f"Press release: {p['former_ceo_of']} names {name} chief executive ({c['from']})",
                    "text": f"Press release: {p['former_ceo_of']} names {name} chief executive ({c['from']})\n\n"
                            f"{p['former_ceo_of']} today named {name} as its new chief executive, effective "
                            f"immediately. {name} said the company would focus on growth in {p['lives']}. "
                            f"This release reflects the situation in {c['from']}."}
            if len(facts) == 1:
                facts.append(f"{name} works as an independent consultant.")
            self._page(name, "person", rng, facts)
        for name, p in self.products.items():
            self._page(name, "product", rng, [
                f"The {name} is {article(p['category'])} {p['category']} made by {p['maker']}.",
                f"It was announced in {p['launched'] - 1} and launched in {p['launched']} at a list price of ${p['price']}."])
        self.pages["wiki/list-of-companies"] = {"title": "List of companies", "text": "List of companies\n\n" + "\n".join(
            f"- {n} ({c['industry']})" for n, c in sorted(self.companies.items()))}
        self.pages["wiki/list-of-products"] = {"title": "List of products", "text": "List of products\n\n" + "\n".join(
            f"- {n} ({p['category']})" for n, p in sorted(self.products.items()))}
        for name, c in self.cities.items():
            metro = c["population"] * rng.randrange(2, 5) + rng.randrange(1000)
            self._page(name, "city", rng, [
                f"{name} is a city in {c['country']}, founded in {c['founded']}.",
                f"The city population is {c['population']:,} (2025 census); the wider metropolitan area has {metro:,} residents."])

    def _page(self, name: str, kind: str, rng, facts: list[str]) -> None:
        sentences = facts[:]
        for _ in range(rng.randrange(14, 32)):
            sentences.insert(rng.randrange(1, len(sentences) + 1),
                             rng.choice(FILLER).format(name=name, kind=kind, place=rng.choice(list(self.cities))))
        self.pages[slug(name)] = {"title": name, "text": f"{name}\n\n" + " ".join(sentences)}

    # ------------------------------------------------------------ questions --

    def _q_two_hop(self, rng):
        if rng.random() < 0.5:
            product = rng.choice(sorted(self.products))
            maker = self.products[product]["maker"]
            return (f"Who is the chief executive of the company that makes the {product}?",
                    self.companies[maker]["ceo"], [product, maker])
        company = rng.choice(sorted(self.companies))
        ceo = self.companies[company]["ceo"]
        return f"In which year was the chief executive of {company} born?", str(self.people[ceo]["born"]), [company, ceo]

    def _q_three_hop(self, rng):
        company = rng.choice(sorted(self.companies))
        ceo = self.companies[company]["ceo"]
        home = self.people[ceo]["lives"]
        return (f"What is the population (city proper, 2025 census) of the city where the chief executive of "
                f"{company} lives?",
                str(self.cities[home]["population"]), [company, ceo, home])

    def _q_argmax(self, rng):
        company = rng.choice(sorted(self.companies))
        city = self.companies[company]["hq"]
        best = max(self.companies[company]["products"], key=lambda p: self.products[p]["launched"])
        return (f"Which product made by the company headquartered in {city} was launched most recently?",
                best, [city, company, *self.companies[company]["products"]])

    def _q_tenure(self, rng):
        # launch years strictly inside a tenure: a handover year would have two answers
        candidates = [(c, p) for c, info in sorted(self.companies.items()) for p in sorted(info["products"])
                      if info["from"] < self.products[p]["launched"] < info["since"]]
        if not candidates or rng.random() < 0.3:
            candidates = [(c, p) for c, info in sorted(self.companies.items()) for p in sorted(info["products"])
                          if self.products[p]["launched"] > info["since"]]
        company, product = rng.choice(candidates)
        info = self.companies[company]
        answer = info["ceo"] if self.products[product]["launched"] >= info["since"] else info["former"]
        return (f"Who was the chief executive of the company that makes the {product} in the year the {product} "
                f"was launched?", answer, [product, company, info["ceo"], info["former"]])

    def _q_aggregate(self, rng):
        births = sorted(self.people[c["ceo"]]["born"] for c in self.companies.values())
        cut = births[rng.randrange(3, 6)]  # 3-5 companies qualify
        chosen = [n for n, c in self.companies.items() if self.people[c["ceo"]]["born"] < cut]
        total = sum(self.products[p]["price"] for n in chosen for p in self.companies[n]["products"])
        return (f"What is the total list price in dollars of all products made by companies whose current chief "
                f"executive was born before {cut}?", str(total),
                ["List of companies", *self.companies, *[self.companies[n]["ceo"] for n in self.companies],
                 *[p for n in chosen for p in self.companies[n]["products"]]])

    # ---------------------------------------------------------------- tools --

    def tool_search(self, query: str) -> str:
        terms = [t for t in norm(query).split() if len(t) > 2]
        if not terms:
            raise ToolError("the query has no searchable terms")
        scored = []
        for page_id, page in self.pages.items():
            title, words = norm(page["title"]), set(norm(page["text"]).split())
            score = sum(3 if t in title.split() else 1 for t in terms if t in words)
            if score:
                scored.append((-score, page_id))
        if not scored:
            return "no matches"
        out = []
        for _, page_id in sorted(scored)[:8]:
            page = self.pages[page_id]
            sentences = re.split(r"(?<=[.!?]) |\n", page["text"].split("\n\n", 1)[1])
            snippet = next((s for s in sentences if any(t in norm(s).split() for t in terms)), sentences[0])
            out.append(f"{page_id} | {page['title']} | {snippet[:200]}")
        return "\n".join(out)

    def tool_open_page(self, page_id: str) -> str:
        page = self.pages.get(page_id.strip()) or self.pages.get(slug(page_id))
        if page is None:
            raise ToolError(f"no page {page_id!r}; use a page id returned by search")
        return page["text"]

    def check(self, final: str) -> tuple[bool, str]:
        if self.submitted is None:
            return False, "no submission"
        got, want = norm(self.submitted.get("answer", "")), norm(self.answer)
        if want.isdigit():
            numbers = re.findall(r"\d+", str(self.submitted.get("answer", "")).replace(",", ""))
            ok = bool(numbers) and numbers[0] == want
        else:
            ok = got == want or (want in got and len(got.split()) <= len(want.split()) + 6)
        return ok, "correct" if ok else f"expected {self.answer!r}"

    def oracle(self) -> list[Step]:
        steps = [Step(calls=[Call("search", {"query": self.path[0]})])]
        steps.append(Step(calls=[Call("open_page", {"page_id": slug(name)}) for name in self.path]))
        steps.append(Step(calls=[Call("submit_answer", {"answer": self.answer})]))
        return steps


def unique_product(rng, taken: set) -> str:
    return synth.unique(rng, lambda: f"{synth.word(rng, 2)} {rng.choice(['X', 'S', 'Pro', 'Mini', 'One', ''])}"
                        f"{rng.randrange(2, 900)}".strip(), taken)
