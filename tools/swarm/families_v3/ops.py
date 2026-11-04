"""Operations (hard tier): precedence policies over a 400-ticket tracker, two teams at once.

Every variant covers two teams in different regions (different public holidays),
has a security override that takes precedence over the normal rule, and comments
that must not be repeated when an interrupted earlier sweep already left them.
Fields such as timestamps, first response, customer tier, component and comments
only show in get_ticket/get_tickets, so the agent reads most of both teams' tickets
(about 200) before it acts, then makes 25-70 changes. Variants: a business-hours
response-SLA sweep (enterprise targets halved, tickets paused while waiting for the
customer); a leave rebalance to the least-loaded qualified engineer under a load cap,
recomputed after every move, with the team lead as the fallback; an ordered
first-match escalation policy whose staleness counts regional business days; a
duplicate merge that resolves chains to their root and folds priority, labels and
status into it. The checker compares the whole tracker, so collateral changes fail.
"""

from __future__ import annotations

import datetime as dt

from families import Call, Step, Task, ToolError, rng_for, synth, tool
from families.ops import NOUNS, VERBS, describe

TEAMS = ["billing", "platform", "mobile", "data"]
REGIONS = {"billing": "UK", "platform": "DE", "mobile": "NL", "data": "IE"}
STATUSES = ["open", "in_progress", "blocked", "resolved", "closed"]
ACTIVE = ("open", "in_progress", "blocked")
PRIORITIES = ["P1", "P2", "P3", "P4"]
SKILLS = ["backend", "frontend", "mobile"]
LABELS = ["customer", "regression", "infra", "ux", "flaky", "docs", "backend", "frontend"]
NOW = dt.datetime(2026, 10, 20, 14, 0)  # a Tuesday
TARGET_HOURS = {"P1": 4, "P2": 8, "P3": 16, "P4": 32}
HOLIDAY_NAMES = ["Founders' Day", "Harvest Monday", "Regional Day", "Bank holiday", "Remembrance Day",
                 "Unity Day", "Civic holiday"]
PAUSE = "Waiting for customer reply."
NOTES = ["Seen again in production this morning.", "Customer escalated via support.",
         "Could be related to the cache rollout.", "Needs a repro on staging.", PAUSE, PAUSE]
PAGE = 25
BATCH = 10
PER_TEAM = 100
KINDS = ["sla", "rebalance", "escalate", "merge"]


def business_deadline(start: dt.datetime, hours: int, holidays: set) -> dt.datetime:
    """start + `hours` business hours, counting 09:00-17:00 Monday-Friday on non-holidays only."""
    t, left = start, hours * 60
    while True:
        day_start = t.replace(hour=9, minute=0)
        day_end = t.replace(hour=17, minute=0)
        if t.weekday() >= 5 or t.date().isoformat() in holidays or t >= day_end:
            t = (t + dt.timedelta(days=1)).replace(hour=9, minute=0)
            continue
        t = max(t, day_start)
        available = int((day_end - t).total_seconds() // 60)
        if left <= available:
            return t + dt.timedelta(minutes=left)
        left -= available
        t = (t + dt.timedelta(days=1)).replace(hour=9, minute=0)


def business_days_back(day: dt.date, n: int, holidays: set) -> dt.date:
    """The business day n business days before `day` (weekends and holidays skipped)."""
    while n:
        day -= dt.timedelta(days=1)
        if day.weekday() < 5 and day.isoformat() not in holidays:
            n -= 1
    return day


class Ops3Task(Task):
    family = "ops3"
    max_turns = 60
    tools = [
        tool("list_tickets", f"List tickets matching all given filters, {PAGE} per page in ticket id order.",
             {"team": {"type": "string", "enum": TEAMS}, "status": {"type": "string", "enum": STATUSES},
              "assignee": {"type": "string"}, "label": {"type": "string"},
              "priority": {"type": "string", "enum": PRIORITIES},
              "page": {"type": "integer", "description": "1-based, default 1"}}, []),
        tool("get_ticket", "Show one ticket in full: timestamps, first response, component, customer tier, "
             "duplicate link and comments.", {"ticket_id": {"type": "string"}}, ["ticket_id"]),
        tool("get_tickets", f"Show up to {BATCH} tickets in full, like get_ticket.",
             {"ticket_ids": {"type": "array", "items": {"type": "string"}}}, ["ticket_ids"]),
        tool("list_users", "List users with team, region, role, skills, on-call, leave and security-champion "
             "status.", {"team": {"type": "string", "enum": TEAMS}}, []),
        tool("list_holidays", "List the public holidays of a region in 2026.",
             {"region": {"type": "string", "enum": sorted(set(REGIONS.values()))}}, ["region"]),
        tool("business_deadline", "Calendar service: the moment `hours` business hours after `start` in a region. "
             "Business hours are 09:00-17:00 Monday to Friday excluding the region's public holidays; a start outside "
             "them counts from the next opening.",
             {"region": {"type": "string", "enum": sorted(set(REGIONS.values()))},
              "start": {"type": "string", "description": "YYYY-MM-DD HH:MM"},
              "hours": {"type": "integer"}}, ["region", "start", "hours"]),
        tool("update_ticket", "Change fields of a ticket. Omitted fields stay unchanged.",
             {"ticket_id": {"type": "string"}, "status": {"type": "string", "enum": STATUSES},
              "assignee": {"type": "string", "description": "A user name from list_users"},
              "priority": {"type": "string", "enum": PRIORITIES},
              "add_labels": {"type": "array", "items": {"type": "string"}},
              "remove_labels": {"type": "array", "items": {"type": "string"}}}, ["ticket_id"]),
        tool("add_comment", "Add a comment to a ticket.", {"ticket_id": {"type": "string"}, "body": {"type": "string"}},
             ["ticket_id", "body"]),
    ]
    latency = {"list_tickets": 0.8, "get_ticket": 0.5, "get_tickets": 1.0, "list_users": 0.5, "list_holidays": 0.4,
               "business_deadline": 0.3, "update_ticket": 0.6, "add_comment": 0.6}

    def __init__(self, index: int, seed: int):
        super().__init__(index, seed)
        rng = rng_for(self.family, seed, index)
        self.kind = KINDS[index % 4]
        self.tier = "hard"
        for _ in range(200):  # regenerate until the policy is substantial
            self._world(rng)
            self.teams = sorted(rng.sample(TEAMS, 2))
            self.before = {t: dict(v, labels=list(v["labels"]), comments=list(v["comments"]))
                           for t, v in self.tickets.items()}
            self.expected = {t: dict(v, labels=list(v["labels"]), comments=list(v["comments"]))
                             for t, v in self.tickets.items()}
            self.tags: dict[str, set] = {}  # requirement -> the tickets it decides
            if getattr(self, "_policy_" + self.kind)(rng) is False:
                continue
            changed = sum(self.expected[t] != self.before[t] for t in self.tickets)
            if 25 <= changed <= 70:
                break
        else:
            raise RuntimeError(f"{self.id}: no well-posed tracker")

    # ------------------------------------------------------------- world --

    def _world(self, rng) -> None:
        taken: set = set()
        self.users = {}
        self.lead, self.absent, self.oncall, self.champion = {}, {}, {}, {}
        for team in TEAMS:
            names = [synth.person(rng, taken) for _ in range(6)]
            lead, engineers = names[0], names[1:]
            absent, oncall, champion = rng.sample(engineers, 3)
            for n in names:
                skills = sorted(rng.sample(SKILLS, rng.choice([1, 1, 2])))
                self.users[n] = {"team": team, "role": "lead" if n == lead else "engineer", "skills": skills,
                                 "on_call": n == oncall, "on_leave": n == absent, "champion": n == champion}
            available = [n for n in engineers if n not in (absent, oncall)]
            for skill in SKILLS:  # every skill has an available engineer
                if not any(skill in self.users[n]["skills"] for n in available):
                    user = self.users[rng.choice(available)]
                    user["skills"] = sorted({*user["skills"], skill})
            self.lead[team], self.absent[team], self.oncall[team], self.champion[team] = lead, absent, oncall, champion
        self.holidays = {}
        window = [NOW.date() - dt.timedelta(days=d) for d in range(1, 45)]
        weekdays = [d for d in window if d.weekday() < 5]
        for region in sorted(set(REGIONS.values())):
            days = rng.sample(weekdays, rng.randrange(2, 5)) + [dt.date(2026, rng.randrange(3, 8), rng.randrange(1, 28))]
            self.holidays[region] = {"2026-01-01": "New Year's Day", "2026-12-25": "Christmas Day",
                                     **{d.isoformat(): rng.choice(HOLIDAY_NAMES) for d in days}}
        self.stale_days = rng.choice([8, 10, 12, 15])
        self.tickets = {}
        for i, num in enumerate(rng.sample(range(10000, 99999), PER_TEAM * len(TEAMS))):
            team = TEAMS[i % 4]
            members = [n for n, u in self.users.items() if u["team"] == team]
            assignee = self.absent[team] if rng.random() < 0.2 else rng.choice(members)
            created = NOW - dt.timedelta(minutes=rng.randrange(30, 60 * 24 * (6 if rng.random() < 0.55 else 40)))
            responded = rng.random() < 0.4
            labels = sorted(rng.sample(LABELS, rng.randrange(0, 3)) + (["security"] if rng.random() < 0.1 else []))
            self.tickets[f"T-{num}"] = {
                "team": team, "title": f"{rng.choice(NOUNS).capitalize()} {rng.choice(VERBS)}",
                "status": rng.choice(STATUSES), "priority": rng.choice(PRIORITIES), "assignee": assignee,
                "labels": labels, "created": created.strftime("%Y-%m-%d %H:%M"),
                "first_response": (created + dt.timedelta(minutes=rng.randrange(5, 600))).strftime("%Y-%m-%d %H:%M")
                if responded else None,
                "component": rng.choice(SKILLS), "tier": rng.choice(["standard", "standard", "enterprise"]),
                "duplicate_of": None, "comments": [], "description": describe(rng)}
        ids = sorted(self.tickets)
        for tid in rng.sample(ids, 90):
            for _ in range(rng.choice([1, 1, 2])):
                self.tickets[tid]["comments"].append(rng.choice(NOTES))
        for team in TEAMS:  # duplicate chains: each duplicate points at a root or an earlier duplicate
            own = [t for t in ids if self.tickets[t]["team"] == team]
            rng.shuffle(own)
            roots, dups = own[:10], own[10:30]
            for k, tid in enumerate(dups):
                t = self.tickets[tid]
                t["labels"] = sorted(set(t["labels"]) | {"duplicate"})
                t["duplicate_of"] = rng.choice(roots + dups[:k][-4:] * 2 if k else roots)

    def _tag(self, tid: str, *names: str) -> None:
        for name in names:
            self.tags.setdefault(name, set()).add(tid)

    def _scope(self):
        return [(tid, t) for tid, t in sorted(self.expected.items()) if t["team"] in self.teams]

    def _who(self) -> str:
        return f"the {self.teams[0]} and {self.teams[1]} teams"

    @staticmethod
    def _comment(t: dict, body: str) -> None:
        """Comments are idempotent: an identical existing comment is not repeated."""
        if body not in t["comments"]:
            t["comments"].append(body)

    def _seed_earlier(self, tid: str, body: str) -> None:
        """An interrupted earlier sweep already left this comment."""
        self._tag(tid, "no_repeated_comment")
        for state in (self.before, self.expected, {tid: self.tickets[tid]}):
            if body not in state[tid]["comments"]:
                state[tid]["comments"].append(body)

    def _security(self, t: dict) -> None:
        t["priority"] = "P1"
        t["assignee"] = self.champion[t["team"]]
        self._comment(t, "Security override")

    IDEMPOTENT = "Never add a comment a ticket already has (an earlier sweep was interrupted). Change nothing else. " \
                 "Reply with a one-line summary when done."
    SECURITY = ("Security override: a matching ticket labelled 'security' instead gets priority P1, is assigned to "
                "its team's security champion and gets the comment 'Security override'.")

    # ---------------------------------------------------------- policies --

    def _policy_sla(self, rng):
        self.prompt = (
            f"{self.header()} Response-SLA sweep for {self._who()}. A ticket's response deadline is its created time "
            f"plus the target for its priority: P1 4, P2 8, P3 16, P4 32 business hours, halved for enterprise "
            f"customers (customer tier, in get_ticket). Work out deadlines with the business_deadline calendar tool "
            f"(business hours, the team's region from list_users, and its public holidays are built into it). Now is "
            f"{NOW.strftime('%Y-%m-%d %H:%M')}. A ticket whose latest comment is '{PAUSE}' is paused and is not "
            f"checked. For every ticket of these teams with status open or in_progress, no first response, not "
            f"paused, and a deadline before now: add the label 'sla-breach', raise its priority to P2 unless it is "
            f"already P1 or P2, and add the comment 'SLA breached: response due YYYY-MM-DD HH:MM' with its deadline. "
            f"{self.SECURITY} {self.IDEMPOTENT}")
        for tid, t in self._scope():
            if t["status"] not in ("open", "in_progress") or t["first_response"]:
                self._tag(tid, "ineligible_untouched")
                continue
            if t["comments"] and t["comments"][-1] == PAUSE:
                self._tag(tid, "paused_untouched")
                continue
            created = dt.datetime.strptime(t["created"], "%Y-%m-%d %H:%M")
            holidays = set(self.holidays[REGIONS[t["team"]]])
            hours = TARGET_HOURS[t["priority"]] // (2 if t["tier"] == "enterprise" else 1)
            deadline = business_deadline(created, hours, holidays)
            if t["tier"] == "enterprise":
                self._tag(tid, "enterprise_target")
            if business_deadline(created, hours, set()) != deadline:
                self._tag(tid, "regional_holidays")
            if created.weekday() >= 5 or not 9 <= created.hour < 17:
                self._tag(tid, "clock_starts_next_business_hour")
            if deadline >= NOW:
                self._tag(tid, "within_sla_untouched")
                continue
            if "security" in t["labels"]:
                self._tag(tid, "security_override")
                if rng.random() < 0.3:
                    self._seed_earlier(tid, "Security override")
                self._security(t)
                continue
            body = f"SLA breached: response due {deadline.strftime('%Y-%m-%d %H:%M')}"
            if rng.random() < 0.25:
                self._seed_earlier(tid, body)
            self._tag(tid, "breach_label_and_comment")
            t["labels"] = sorted(set(t["labels"]) | {"sla-breach"})
            if t["priority"] in ("P3", "P4"):
                self._tag(tid, "priority_raised_to_p2")
                t["priority"] = "P2"
            self._comment(t, body)

    def _loads(self, team: str) -> dict:
        load = {n: 0 for n, u in self.users.items() if u["team"] == team}
        for t in self.expected.values():
            if t["status"] in ACTIVE and t["assignee"] in load:
                load[t["assignee"]] += 1
        return load

    def _policy_rebalance(self, rng):
        a, b = self.teams
        floors = []
        for team in self.teams:  # the cap binds part-way through each team's reassignments
            load = self._loads(team)
            floors.append(min(load[n] for n, u in self.users.items() if u["team"] == team and u["role"] == "engineer"
                              and not u["on_leave"] and not u["on_call"]))
        self.cap = max(floors) + rng.randrange(3, 6)
        self.prompt = (
            f"{self.header()} Leave rebalance for {self._who()}: {self.absent[a]} ({a}) and {self.absent[b]} ({b}) "
            f"are on leave. For each team, reassign the person on leave's tickets with status open, in_progress or "
            f"blocked one at a time in ascending ticket id order. A ticket goes to the team's engineer who has the "
            f"ticket's component (get_ticket) among their skills, is neither on leave nor on call, is not the team "
            f"lead, holds fewer than {self.cap} active tickets (status open, in_progress or blocked) at that moment, "
            f"and of those holds the fewest, counting the reassignments already made; break ties by name, "
            f"alphabetically. If no engineer qualifies, the ticket goes to the team lead. {self.SECURITY.replace('a matching', 'a')} "
            f"Add the comment 'Reassigned from <name> (on leave)' to every ticket you reassign. Leave resolved and "
            f"closed tickets alone. {self.IDEMPOTENT}")
        fallback = assigned = 0
        for team in self.teams:
            absent = self.absent[team]
            load = self._loads(team)
            for tid, t in self._scope():
                if t["team"] != team:
                    continue
                if t["assignee"] != absent or t["status"] not in ACTIVE:
                    self._tag(tid, "untouched")
                    continue
                if "security" in t["labels"]:
                    self._tag(tid, "security_override")
                    target = self.champion[team]
                    t["priority"] = "P1"
                    self._comment(t, "Security override")
                else:
                    skilled = [n for n, u in self.users.items() if u["team"] == team and u["role"] == "engineer"
                               and not u["on_leave"] and not u["on_call"] and t["component"] in u["skills"]]
                    eligible = [n for n in skilled if load[n] < self.cap]
                    target = min(eligible, key=lambda n: (load[n], n)) if eligible else self.lead[team]
                    fallback += not eligible
                    assigned += bool(eligible)
                    self._tag(tid, "least_loaded_qualified" if eligible else "lead_fallback_at_cap")
                    if len(eligible) < len(skilled):
                        self._tag(tid, "load_cap")
                    if eligible and sum(load[n] == load[target] for n in eligible) > 1:
                        self._tag(tid, "tie_break_by_name")
                t["assignee"] = target
                load[absent] -= 1
                load[target] += 1
                self._comment(t, f"Reassigned from {absent} (on leave)")
        return fallback >= 3 and assigned >= 6  # the cap binds, but not from the start

    def _stale_day(self, team: str) -> dt.date:
        return business_days_back(NOW.date(), self.stale_days, set(self.holidays[REGIONS[team]]))

    def _policy_escalate(self, rng):
        self.prompt = (
            f"{self.header()} Escalation sweep for {self._who()}. A ticket is stale if it was created before the day "
            f"{self.stale_days} business days before today ({NOW.date().isoformat()}), counting back Monday to "
            f"Friday and skipping the public holidays of the ticket's team region (list_users, list_holidays). Apply "
            f"exactly one rule to every ticket of these teams with status open, in_progress or blocked: the first "
            f"rule that matches, in this order. "
            f"(1) Labelled 'security' and not blocked: priority P1, assign it to its team's security champion, "
            f"comment 'Security override'. "
            f"(2) Blocked and stale: only the comment 'Escalation skipped: blocked'. "
            f"(3) Priority P1 or P2, stale, customer tier enterprise (get_ticket): assign it to its team lead, add "
            f"the label 'escalated' and the comment 'Escalated to lead'. "
            f"(4) Priority P1 or P2 and stale: add the label 'escalated' only. "
            f"(5) Priority P3, stale and labelled 'customer': set priority P2 and add the label 'escalated'. "
            f"Tickets matching no rule stay as they are. {self.IDEMPOTENT}")
        for tid, t in self._scope():
            if t["status"] not in ACTIVE:
                self._tag(tid, "untouched")
                continue
            stale = t["created"][:10] < self._stale_day(t["team"]).isoformat()
            plain = t["created"][:10] < business_days_back(NOW.date(), self.stale_days, set()).isoformat()
            if stale != plain:
                self._tag(tid, "regional_business_days")
            rule = ("1_security" if "security" in t["labels"] and t["status"] != "blocked" else
                    "2_blocked" if t["status"] == "blocked" and stale else
                    "3_enterprise_to_lead" if t["priority"] in ("P1", "P2") and stale and t["tier"] == "enterprise" else
                    "4_escalate" if t["priority"] in ("P1", "P2") and stale else
                    "5_customer_p3" if t["priority"] == "P3" and stale and "customer" in t["labels"] else
                    "no_rule_untouched")
            self._tag(tid, "rule" + rule if rule[0].isdigit() else rule)
            if "security" in t["labels"] and t["status"] != "blocked":
                if rng.random() < 0.3:
                    self._seed_earlier(tid, "Security override")
                self._security(t)
            elif t["status"] == "blocked" and stale:
                if rng.random() < 0.3:
                    self._seed_earlier(tid, "Escalation skipped: blocked")
                self._comment(t, "Escalation skipped: blocked")
            elif t["priority"] in ("P1", "P2") and stale and t["tier"] == "enterprise":
                if rng.random() < 0.3:
                    self._seed_earlier(tid, "Escalated to lead")
                t["assignee"] = self.lead[t["team"]]
                t["labels"] = sorted(set(t["labels"]) | {"escalated"})
                self._comment(t, "Escalated to lead")
            elif t["priority"] in ("P1", "P2") and stale:
                t["labels"] = sorted(set(t["labels"]) | {"escalated"})
            elif t["priority"] == "P3" and stale and "customer" in t["labels"]:
                t["priority"] = "P2"
                t["labels"] = sorted(set(t["labels"]) | {"escalated"})

    def _root(self, tid: str) -> str:
        seen = set()
        while self.before[tid]["duplicate_of"] and tid not in seen:
            seen.add(tid)
            tid = self.before[tid]["duplicate_of"]
        return tid

    def _policy_merge(self, rng):
        self.prompt = (
            f"{self.header()} Merge duplicates for {self._who()}. A duplicate's 'Duplicate of' link (get_ticket) can "
            f"point at another duplicate; follow the links to the root, the first ticket without one. For every "
            f"ticket of these teams labelled 'duplicate' whose status is not closed: if it is labelled 'security', "
            f"do not close it; only add the comment 'Security duplicate: needs review'. Otherwise close it with the "
            f"comment 'Closed as duplicate of <root id>'. Then, for every root that had duplicates closed in this "
            f"sweep: raise its priority to the highest priority among itself and those duplicates (P1 is highest, "
            f"using priorities from before the sweep); add the label 'customer' if any of those duplicates has it; "
            f"and if the root is resolved or closed, set it to open and add the comment 'Reopened: duplicate "
            f"reports'. {self.IDEMPOTENT}")
        merged: dict[str, list[str]] = {}
        for tid, t in self._scope():
            if "duplicate" not in t["labels"] or t["status"] == "closed":
                continue
            if "security" in t["labels"]:
                self._tag(tid, "security_duplicate")
                if rng.random() < 0.4:
                    self._seed_earlier(tid, "Security duplicate: needs review")
                self._comment(t, "Security duplicate: needs review")
                continue
            root = self._root(tid)
            self._tag(tid, "close_with_root_comment")
            if root != t["duplicate_of"]:
                self._tag(tid, "chain_followed_to_root")
            t["status"] = "closed"
            self._comment(t, f"Closed as duplicate of {root}")
            merged.setdefault(root, []).append(tid)
        for root, dups in merged.items():
            r = self.expected[root]
            r["priority"] = min([self.before[root]["priority"]] + [self.before[d]["priority"] for d in dups])
            if r["priority"] != self.before[root]["priority"]:
                self._tag(root, "root_priority_raised")
            if any("customer" in self.before[d]["labels"] for d in dups):
                if "customer" not in r["labels"]:
                    self._tag(root, "root_customer_label")
                r["labels"] = sorted(set(r["labels"]) | {"customer"})
            if self.before[root]["status"] in ("resolved", "closed"):
                self._tag(root, "root_reopened")
                r["status"] = "open"
                self._comment(r, "Reopened: duplicate reports")

    # ------------------------------------------------------------- tools --

    def _ticket(self, ticket_id: str) -> dict:
        t = self.tickets.get(ticket_id.strip().upper())
        if t is None:
            raise ToolError(f"no ticket {ticket_id!r}")
        return t

    def tool_list_tickets(self, team: str = "", status: str = "", assignee: str = "", label: str = "",
                          priority: str = "", page: int = 1) -> str:
        rows = [f"{tid} | {t['team']} | {t['priority']} | {t['status']} | {t['assignee']} | "
                f"labels: {', '.join(t['labels']) or '-'} | {t['title']}"
                for tid, t in sorted(self.tickets.items())
                if (not team or t["team"] == team) and (not status or t["status"] == status)
                and (not assignee or t["assignee"].lower() == assignee.strip().lower())
                and (not label or label.strip().lower() in t["labels"]) and (not priority or t["priority"] == priority)]
        if not rows:
            return "(none)"
        pages = (len(rows) + PAGE - 1) // PAGE
        if not 1 <= page <= pages:
            raise ToolError(f"page must be between 1 and {pages}")
        return "\n".join([f"page {page} of {pages} ({len(rows)} tickets)",
                          "id | team | priority | status | assignee | labels | title"]
                         + rows[(page - 1) * PAGE:page * PAGE])

    def tool_get_ticket(self, ticket_id: str) -> str:
        t = self._ticket(ticket_id)
        lines = [f"{ticket_id.strip().upper()}: {t['title']}", f"Team: {t['team']}", f"Status: {t['status']}",
                 f"Priority: {t['priority']}", f"Assignee: {t['assignee']}", f"Created: {t['created']}",
                 f"First response: {t['first_response'] or '-'}", f"Component: {t['component']}",
                 f"Customer tier: {t['tier']}", f"Labels: {', '.join(t['labels']) or '-'}",
                 f"Duplicate of: {t['duplicate_of'] or '-'}", f"Description: {t['description']}", "Comments:"]
        lines += [f"  - {c}" for c in t["comments"]] or ["  (none)"]
        return "\n".join(lines)

    def tool_get_tickets(self, ticket_ids: list) -> str:
        if not 1 <= len(ticket_ids) <= BATCH:
            raise ToolError(f"give between 1 and {BATCH} ticket ids")
        return "\n\n".join(self.tool_get_ticket(str(i)) for i in ticket_ids)

    def tool_list_users(self, team: str = "") -> str:
        yn = lambda b: "yes" if b else "no"  # noqa: E731
        rows = [f"{n} | {u['team']} | {REGIONS[u['team']]} | {u['role']} | {', '.join(u['skills'])} | "
                f"{yn(u['on_call'])} | {yn(u['on_leave'])} | {yn(u['champion'])}"
                for n, u in sorted(self.users.items()) if not team or u["team"] == team]
        return "name | team | region | role | skills | on call | on leave | security champion\n" + "\n".join(rows)

    def tool_list_holidays(self, region: str) -> str:
        return "\n".join(f"{d} {name}" for d, name in sorted(self.holidays[region].items()))

    def tool_business_deadline(self, region: str, start: str, hours: int) -> str:
        try:
            when = dt.datetime.strptime(start.strip(), "%Y-%m-%d %H:%M")
        except ValueError:
            raise ToolError("start must look like 2026-10-19 16:30") from None
        if hours < 0:
            raise ToolError("hours must not be negative")
        return business_deadline(when, hours, set(self.holidays[region])).strftime("%Y-%m-%d %H:%M")

    def tool_update_ticket(self, ticket_id: str, status: str = "", assignee: str = "", priority: str = "",
                           add_labels: list | None = None, remove_labels: list | None = None) -> str:
        t = self._ticket(ticket_id)
        if assignee:
            match = next((n for n in self.users if n.lower() == assignee.strip().lower()), None)
            if match is None:
                raise ToolError(f"unknown user {assignee!r}")
            t["assignee"] = match
        if status:
            t["status"] = status
        if priority:
            t["priority"] = priority
        labels = set(t["labels"]) | {lb.strip().lower() for lb in add_labels or []}
        t["labels"] = sorted(labels - {lb.strip().lower() for lb in remove_labels or []})
        return f"Updated {ticket_id.strip().upper()}."

    def tool_add_comment(self, ticket_id: str, body: str) -> str:
        self._ticket(ticket_id)["comments"].append(body)
        return f"Comment added to {ticket_id.strip().upper()}."

    # ----------------------------------------------------------- checker --

    def check(self, final: str) -> tuple[bool, str]:
        wrong, bad = [], set()
        norm = lambda cs: [" ".join(c.strip().rstrip(".").lower().split()) for c in cs]  # noqa: E731
        for tid, want in self.expected.items():
            have = self.tickets[tid]
            for key in ("status", "assignee", "priority"):
                if have[key] != want[key]:
                    wrong.append(f"{tid}.{key}")
            if sorted(have["labels"]) != sorted(want["labels"]):
                wrong.append(f"{tid}.labels")
            if sorted(norm(have["comments"])) != sorted(norm(want["comments"])):
                wrong.append(f"{tid}.comments")
            if wrong and wrong[-1].startswith(tid + "."):
                bad.add(tid)
        outside = {tid for tid, t in self.before.items() if t["team"] not in self.teams}
        counts = lambda tids: [len(tids - bad), len(tids)]  # noqa: E731  [tickets right, tickets the rule decides]
        self.requirements = {f"{self.kind}.{name}": counts(tids) for name, tids in sorted(self.tags.items())}
        self.requirements[f"{self.kind}.other_teams_untouched"] = counts(outside)
        return not wrong, "tracker state matches" if not wrong else f"{len(wrong)} wrong: " + ", ".join(wrong[:8])

    def oracle(self) -> list[Step]:
        scope = [tid for tid, t in sorted(self.before.items()) if t["team"] in self.teams]
        updates, comments = [], []
        for tid, want in sorted(self.expected.items()):
            have = self.before[tid]
            args = {k: want[k] for k in ("status", "assignee", "priority") if want[k] != have[k]}
            added = sorted(set(want["labels"]) - set(have["labels"]))
            if added:
                args["add_labels"] = added
            if args:
                updates.append(Call("update_ticket", {"ticket_id": tid, **args}))
            for body in want["comments"][len(have["comments"]):]:
                comments.append(Call("add_comment", {"ticket_id": tid, "body": body}))
        return [Step(calls=[Call("list_users", {"team": team}) for team in self.teams]
                     + [Call("list_holidays", {"region": REGIONS[team]}) for team in self.teams]),
                Step(calls=[Call("list_tickets", {"team": team, "page": p}) for team in self.teams for p in (1, 2, 3, 4)]),
                Step(calls=[Call("get_tickets", {"ticket_ids": scope[i:i + BATCH]}) for i in range(0, len(scope), BATCH)]),
                Step(calls=updates + comments),
                Step(content=f"Done: {len(updates)} updates, {len(comments)} comments.")]
