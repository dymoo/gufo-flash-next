"""Synthetic names, boilerplate and agent handbooks.

Invented names keep tasks out of pretraining data; templated boilerplate gives
documents and system prompts realistic length without depending on any corpus.
"""

from __future__ import annotations

CHARS_PER_TOKEN = 3.8  # English prose, as tools/qwen-flash/qa/agent_sim.py

ONSETS = ["b", "br", "c", "d", "dr", "f", "g", "h", "k", "kr", "l", "m", "n", "p", "r", "s", "st", "t", "tr", "v",
          "w", "z", "th", "sh", "gl", "pl", "qu"]
VOWELS = ["a", "e", "i", "o", "u", "ae", "ia", "ou", "ei", "y"]
CODAS = ["", "", "n", "r", "l", "s", "x", "nd", "rk", "lt", "m", "v", "th", "st"]
INDUSTRY = ["Dynamics", "Systems", "Logistics", "Labs", "Industries", "Works", "Analytics", "Foods", "Robotics",
            "Materials", "Networks", "Optics", "Pharma", "Energy", "Textiles", "Instruments"]
LEGAL = ["GmbH", "Ltd", "Inc.", "S.A.", "B.V.", "AB", "Oy", "S.r.l."]
CITY_PREFIX = ["", "", "", "Port ", "Saint ", "New ", "Upper ", "Lake "]
STREET = ["Lane", "Road", "Street", "Avenue", "Way", "Quay", "Row"]


def word(rng, syllables: int) -> str:
    return "".join(rng.choice(ONSETS) + rng.choice(VOWELS) + rng.choice(CODAS) for _ in range(syllables)).capitalize()


def unique(rng, make, taken: set) -> str:
    while True:
        value = make()
        if value.lower() not in taken:
            taken.add(value.lower())
            return value


def company(rng, taken: set) -> str:
    return unique(rng, lambda: f"{word(rng, 2)} {rng.choice(INDUSTRY)}", taken)


def person(rng, taken: set) -> str:
    return unique(rng, lambda: f"{word(rng, rng.choice([2, 2, 3]))} {word(rng, rng.choice([2, 3]))}", taken)


def city(rng, taken: set) -> str:
    return unique(rng, lambda: rng.choice(CITY_PREFIX) + word(rng, rng.choice([2, 3])), taken)


def street(rng) -> str:
    return f"{rng.randrange(1, 240)} {word(rng, 2)} {rng.choice(STREET)}"


BOILER_SUBJECTS = ["The Supplier", "The Customer", "Either party", "The carrier", "The receiving department",
                   "The account holder", "The issuing office", "Any authorised representative"]
BOILER_ACTIONS = ["shall notify the other party in writing", "must retain the original documentation",
                  "may suspend further deliveries", "shall bear the cost of return shipping",
                  "is entitled to statutory interest", "must report visible damage", "shall issue a corrected document",
                  "may request a certificate of conformity", "shall not assign this agreement",
                  "must quote the document number on all correspondence"]
BOILER_CONDITIONS = ["within {n} days of receipt", "unless otherwise agreed in writing",
                     "where goods are found to be non-conforming", "after the agreed delivery window has passed",
                     "if payment is received after the due date", "for orders above {n} units",
                     "in the event of force majeure", "before any claim is accepted"]
BOILER_TAILS = ["", " Claims raised later cannot be considered.", " This does not limit statutory rights.",
                " Reference regulation {n}/{m} applies.", " Partial deliveries are invoiced separately.",
                " Records are kept for {n} years.", " Exceptions require approval by a director."]


def boilerplate(rng, words: int) -> str:
    """Contract-like filler of roughly `words` words, in numbered clauses."""
    out, count, clause = [], 0, 1
    while count < words:
        text = (f"{clause}. {rng.choice(BOILER_SUBJECTS)} {rng.choice(BOILER_ACTIONS)} "
                f"{rng.choice(BOILER_CONDITIONS)}.{rng.choice(BOILER_TAILS)}")
        text = text.format(n=rng.randrange(2, 90), m=rng.randrange(10, 99))
        out.append(text)
        count += len(text.split())
        clause += 1
    return "\n".join(out)


HANDBOOK = {
    "code": ("software engineering agent working in a Python repository through tools",
             ["Repository hygiene", "Testing discipline", "Code review expectations", "Dependency policy",
              "Error handling", "Naming and style", "Performance budgets", "Documentation", "Release process",
              "Security review", "Logging", "Backward compatibility"],
             ["changing a public function", "a test fails intermittently", "touching shared helpers",
              "adding a dependency", "a function grows past fifty lines", "an error is swallowed",
              "a docstring disagrees with behaviour", "fixing a regression", "editing configuration files",
              "handling user input", "writing a migration", "a change spans several modules"],
             ["keep the change minimal and focused", "prefer explicit return values over sentinels",
              "describe the reason in the summary", "keep existing behaviour for unrelated callers",
              "run the full suite before finishing", "avoid broad exception handlers",
              "match the surrounding style", "do not reformat untouched code", "keep functions pure where practical",
              "document edge cases in the docstring", "prefer the standard library",
              "never weaken an existing assertion"]),
    "extract": ("back-office data extraction agent that reads business documents through tools",
                ["Source documents", "Dates and times", "Currencies and amounts", "Vendor identity", "Line items",
                 "Duplicates", "Corrections", "Data protection", "Audit trail", "Rounding", "Escalation",
                 "Record formats"],
                ["a document shows several dates", "an amount appears in more than one place",
                 "a vendor name has a legal suffix", "figures use a European decimal comma",
                 "a document is marked as a copy", "a total disagrees with its lines", "a field is illegible",
                 "a document references another document", "a currency symbol is ambiguous",
                 "a value is crossed out", "a credit note is attached", "the customer is named"],
                ["record values exactly as they appear after normalisation", "use ISO 8601 dates",
                 "never infer a value that the document does not support", "keep two decimals for amounts",
                 "prefer the explicitly labelled field", "keep personal data out of free-text notes",
                 "submit one record per request", "use the legal entity name of the issuer",
                 "follow the schema of the submission tool", "leave optional notes empty",
                 "cross-check totals before submitting", "quote document numbers verbatim"]),
    "research": ("research agent answering questions from an internal encyclopedia through search tools",
                 ["Source quality", "Search strategy", "Entity resolution", "Numbers and dates", "Citations",
                  "Ambiguity", "Time-sensitive facts", "Answer format", "Conflicts between pages", "Scope",
                  "Bias", "Verification"],
                 ["two pages disagree", "a name matches several entities", "a question needs several hops",
                  "a page is a stub", "a figure is approximate", "a search returns nothing useful",
                  "a fact depends on a date", "a page mentions a former role", "a list is incomplete",
                  "a question compares entities", "an answer is a number", "an alias is used"],
                 ["open the page before relying on a snippet", "prefer the entity's own page",
                  "verify each hop independently", "answer with the exact name as written on its page",
                  "state numbers without units unless asked", "search again with different keywords",
                  "do not guess when a page is missing", "keep the final answer short",
                  "resolve aliases to canonical names", "compare values only after collecting all of them",
                  "check dates against the question", "submit exactly one answer"]),
    "ops": ("operations agent maintaining the team issue tracker through tools",
            ["Ticket lifecycle", "Assignment", "Priorities", "Labels", "Comments", "On-call", "Leave and absence",
             "Escalation", "Duplicates", "Blocked work", "Auditability", "Bulk changes"],
            ["a ticket changes owner", "an assignee is absent", "a ticket is a duplicate", "work is blocked",
             "a priority changes", "several tickets need the same change", "a ticket is reopened",
             "a label is missing", "a comment is required", "the on-call rota changes",
             "a ticket belongs to another team", "an instruction is ambiguous"],
            ["change only the tickets the instruction covers", "use the exact comment text requested",
             "verify the current state before updating", "keep labels in lowercase",
             "never close a ticket that the instruction does not cover", "prefer one update per ticket",
             "check team membership before reassigning", "leave unrelated fields unchanged",
             "record the reason in a comment only when asked", "re-read a ticket after a failed update",
             "finish with a short summary of the changes", "do not create new labels unless asked"]),
}
REASONS = ["so that reviewers can follow the change", "because downstream systems depend on it",
           "to keep the audit trail complete", "to avoid surprising other teams", "because it reduces rework",
           "as required by the internal controls", "so results stay reproducible", "to limit the blast radius",
           "because past incidents started this way", "to keep the history readable"]


def handbook(kind: str, tokens: int, rng) -> str:
    """A long, internally consistent policy handbook for an agent type."""
    role, sections, situations, directives = HANDBOOK[kind]
    parts, size, n = [], 0, 0
    budget = tokens * CHARS_PER_TOKEN
    while size < budget:
        title = sections[n % len(sections)] + ("" if n < len(sections) else f" (part {n // len(sections) + 1})")
        rules = [f"{n + 1}.{j + 1} When {rng.choice(situations)}, {rng.choice(directives)}, {rng.choice(REASONS)}."
                 for j in range(rng.randrange(5, 9))]
        block = f"### {n + 1}. {title}\n\n" + "\n".join(rules)
        parts.append(block)
        size += len(block) + 2
        n += 1
    return role, "\n\n".join(parts)


def system_prompt(kind: str, tokens: int, rng) -> str:
    role, body = handbook(kind, tokens, rng)
    return (f"You are a {role}. You work autonomously: call the provided tools to inspect and change state, "
            "and do not ask the user questions. Tool results are returned to you as messages. Keep replies brief. "
            "The ticket in the user message is your task; the handbook below is background guidance, and the "
            "ticket's instructions take precedence over it.\n\n## Handbook\n\n" + body + "\n")
