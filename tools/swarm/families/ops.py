"""Operations agent: apply a policy to an issue tracker through tools.

The checker compares the final tracker state with the state the policy implies,
so missed tickets, collateral changes and wrong comments all fail. Tiers: close
duplicates with a reference comment (easy); reassign an absent engineer's open
tickets to the team's on-call (medium); escalate stale high-priority tickets with
an exception for blocked work, or an SLA sweep whose due dates and customer tiers
only get_ticket shows (hard). Tool-call heavy: one update per ticket.
"""

from __future__ import annotations

import datetime as dt

from . import Call, Step, Task, ToolError, rng_for, synth, tool

TEAMS = ["billing", "platform", "mobile", "data"]
STATUSES = ["open", "in_progress", "blocked", "resolved", "closed"]
PRIORITIES = ["P1", "P2", "P3", "P4"]
LABELS = ["customer", "regression", "infra", "ux", "security", "flaky", "docs", "duplicate", "backend", "frontend"]
VERBS = ["fails", "times out", "shows wrong totals", "crashes", "is slow", "returns 500", "drops events",
         "double-charges", "loses state", "renders blank"]
NOUNS = ["export job", "login page", "invoice PDF", "search index", "push sender", "sync worker", "refund flow",
         "dashboard", "webhook relay", "report builder", "cache warmer", "upload API"]
TODAY = dt.date(2026, 10, 20)
STEPS = ["open the {n} for a large account", "retry the {n} after a timeout", "run the {n} with an empty filter",
         "switch the {n} to the EU region", "export the {n} as CSV", "load the {n} on a slow connection"]
SEEN = ["Started after the last release.", "Intermittent; about one request in twenty.", "Only on the mobile app.",
        "Reproduced on staging.", "Several customers wrote in about it.", "Workaround: refresh twice.",
        "Logs show a timeout from the downstream service.", "Not seen before the cache change."]


def describe(rng) -> str:
    noun = rng.choice(NOUNS)
    return (f"Steps: {rng.choice(STEPS).format(n=noun)}, then {rng.choice(STEPS).format(n=rng.choice(NOUNS))}. "
            f"Expected the {noun} to finish normally; it {rng.choice(VERBS)} instead. "
            + " ".join(rng.sample(SEEN, rng.randrange(1, 4))))


class OpsTask(Task):
    family = "ops"
    max_turns = 40
    tools = [
        tool("list_tickets", "List tickets matching all given filters, one line each.",
             {"team": {"type": "string", "enum": TEAMS}, "status": {"type": "string", "enum": STATUSES},
              "assignee": {"type": "string"}, "label": {"type": "string"}}, []),
        tool("get_ticket", "Show one ticket in full, including its comments.", {"ticket_id": {"type": "string"}},
             ["ticket_id"]),
        tool("list_users", "List users with team, role, on-call and leave status.",
             {"team": {"type": "string", "enum": TEAMS}}, []),
        tool("update_ticket", "Change fields of a ticket. Omitted fields stay unchanged.",
             {"ticket_id": {"type": "string"}, "status": {"type": "string", "enum": STATUSES},
              "assignee": {"type": "string", "description": "A user name from list_users"},
              "priority": {"type": "string", "enum": PRIORITIES},
              "add_labels": {"type": "array", "items": {"type": "string"}},
              "remove_labels": {"type": "array", "items": {"type": "string"}}}, ["ticket_id"]),
        tool("add_comment", "Add a comment to a ticket.", {"ticket_id": {"type": "string"}, "body": {"type": "string"}},
             ["ticket_id", "body"]),
    ]
    latency = {"list_tickets": 0.8, "get_ticket": 0.5, "list_users": 0.5, "update_ticket": 0.6, "add_comment": 0.6}

    def __init__(self, index: int, seed: int):
        super().__init__(index, seed)
        rng = rng_for(self.family, seed, index)
        kind = ["duplicates", "reassign", "escalate", "sla"][index % 4]
        self.kind = kind
        self.tier = {"duplicates": "easy", "reassign": "medium"}.get(kind, "hard")
        while True:  # regenerate until the policy changes at least two tickets
            self._tracker(rng)
            self.team = rng.choice(TEAMS)
            self._policy(kind, self.team)
            if sum(self.expected[t] != self.before[t] for t in self.tickets) >= 2:
                break

    def _policy(self, kind: str, team: str) -> None:
        self.before = {t: dict(v, labels=list(v["labels"]), comments=list(v["comments"])) for t, v in self.tickets.items()}
        self.expected = {t: dict(v, labels=list(v["labels"]), comments=list(v["comments"])) for t, v in self.tickets.items()}
        header = self.header()
        if kind == "duplicates":
            self.prompt = (f"{header} Clean up duplicates for the {team} team: every {team} ticket labelled "
                           f"'duplicate' that is not already closed must be closed, with the comment "
                           f"'Closed as duplicate of <id>' where <id> is the ticket named in its 'Duplicate of' field. "
                           f"Change nothing else. Reply with a one-line summary when done.")
            for tid, t in self.expected.items():
                if t["team"] == team and "duplicate" in t["labels"] and t["status"] != "closed":
                    t["status"] = "closed"
                    t["comments"].append(f"Closed as duplicate of {t['duplicate_of']}")
        elif kind == "reassign":
            absent = self.absent[team]
            oncall = self.oncall[team]
            self.prompt = (f"{header} {absent} from the {team} team is on leave. Reassign every {team} ticket assigned "
                           f"to them whose status is open, in_progress or blocked to the {team} team's current "
                           f"on-call engineer (see list_users), and add the comment 'Reassigned: {absent} on leave' "
                           f"to each ticket you reassign. Leave resolved and closed tickets alone. Change nothing "
                           f"else. Reply with a one-line summary when done.")
            for tid, t in self.expected.items():
                if t["team"] == team and t["assignee"] == absent and t["status"] in ("open", "in_progress", "blocked"):
                    t["assignee"] = oncall
                    t["comments"].append(f"Reassigned: {absent} on leave")
        elif kind == "sla":
            lead = self.lead[team]
            self.prompt = (f"{header} SLA sweep for the {team} team. Every {team} ticket with status open or "
                           f"in_progress whose SLA due date (shown by get_ticket) is before {TODAY.isoformat()} must get "
                           f"priority P1 and the label 'sla-breach'. If such a ticket's customer tier is enterprise, "
                           f"also assign it to the {team} team lead (see list_users) and add the comment 'Enterprise "
                           f"SLA breach escalated'. Change nothing else. Reply with a one-line summary when done.")
            for tid, t in self.expected.items():
                if t["team"] == team and t["status"] in ("open", "in_progress") and t["sla_due"] < TODAY.isoformat():
                    t["priority"] = "P1"
                    if "sla-breach" not in t["labels"]:
                        t["labels"].append("sla-breach")
                    if t["tier"] == "enterprise":
                        t["assignee"] = lead
                        t["comments"].append("Enterprise SLA breach escalated")
        else:
            cutoff = TODAY - dt.timedelta(days=self.stale_days)
            lead = self.lead[team]
            self.prompt = (f"{header} Escalation sweep for the {team} team. For every {team} ticket with priority P1 "
                           f"or P2, status open or in_progress, created before {cutoff.isoformat()}: set priority "
                           f"P1, add the label 'escalated' and assign it to the {team} team lead (see list_users). "
                           f"Tickets that match the same criteria but have status blocked must instead get only the "
                           f"comment 'Escalation skipped: blocked'. Change nothing else. Reply with a one-line "
                           f"summary when done.")
            for tid, t in self.expected.items():
                if t["team"] != team or t["priority"] not in ("P1", "P2") or t["created"] >= cutoff.isoformat():
                    continue
                if t["status"] in ("open", "in_progress"):
                    t["priority"] = "P1"
                    t["assignee"] = lead
                    if "escalated" not in t["labels"]:
                        t["labels"].append("escalated")
                elif t["status"] == "blocked":
                    t["comments"].append("Escalation skipped: blocked")

    def _tracker(self, rng) -> None:
        taken: set = set()
        self.users, self.absent, self.oncall, self.lead = {}, {}, {}, {}
        for team in TEAMS:
            names = [synth.person(rng, taken) for _ in range(4)]
            roles = ["lead", "engineer", "engineer", "engineer"]
            oncall = rng.choice(names[1:])
            absent = rng.choice([n for n in names[1:] if n != oncall])
            for n, role in zip(names, roles):
                self.users[n] = {"team": team, "role": role, "on_call": n == oncall, "on_leave": n == absent}
            self.absent[team], self.oncall[team], self.lead[team] = absent, oncall, names[0]
        self.stale_days = rng.choice([14, 21, 30])
        self.tickets = {}
        ids = rng.sample(range(1000, 9999), 120)
        for i, num in enumerate(ids):
            team = TEAMS[i % 4]
            members = [n for n, u in self.users.items() if u["team"] == team]
            assignee = self.absent[team] if rng.random() < 0.3 else rng.choice(members)
            labels = sorted(rng.sample([lb for lb in LABELS if lb != "duplicate"], rng.randrange(0, 3)))
            ticket = {"team": team, "title": f"{rng.choice(NOUNS).capitalize()} {rng.choice(VERBS)}",
                      "status": rng.choice(STATUSES), "priority": rng.choice(PRIORITIES), "assignee": assignee,
                      "labels": labels, "created": (TODAY - dt.timedelta(days=rng.randrange(1, 90))).isoformat(),
                      "duplicate_of": None, "comments": [],
                      "sla_due": (TODAY + dt.timedelta(days=rng.randrange(-20, 30))).isoformat(),
                      "tier": rng.choice(["standard", "standard", "enterprise"]),
                      "description": describe(rng)}
            self.tickets[f"T-{num}"] = ticket
        ids = sorted(self.tickets)
        for tid in rng.sample(ids, 26):  # duplicates point at another ticket of the same team
            t = self.tickets[tid]
            t["labels"] = sorted(set(t["labels"]) | {"duplicate"})
            t["duplicate_of"] = rng.choice([o for o in ids if o != tid and self.tickets[o]["team"] == t["team"]])
        for tid in rng.sample(ids, 24):
            self.tickets[tid]["comments"].append(rng.choice(
                ["Seen again in production this morning.", "Customer escalated via support.",
                 "Could be related to the cache rollout.", "Needs a repro on staging."]))

    # ---------------------------------------------------------------- tools --

    def _ticket(self, ticket_id: str) -> dict:
        t = self.tickets.get(ticket_id.strip().upper())
        if t is None:
            raise ToolError(f"no ticket {ticket_id!r}")
        return t

    def tool_list_tickets(self, team: str = "", status: str = "", assignee: str = "", label: str = "") -> str:
        rows = [f"{tid} | {t['team']} | {t['priority']} | {t['status']} | {t['assignee']} | created {t['created']} | "
                f"labels: {', '.join(t['labels']) or '-'} | {t['title']}"
                for tid, t in sorted(self.tickets.items())
                if (not team or t["team"] == team) and (not status or t["status"] == status)
                and (not assignee or t["assignee"].lower() == assignee.strip().lower())
                and (not label or label.strip().lower() in t["labels"])]
        return "id | team | priority | status | assignee | created | labels | title\n" + ("\n".join(rows) or "(none)")

    def tool_get_ticket(self, ticket_id: str) -> str:
        t = self._ticket(ticket_id)
        lines = [f"{ticket_id.strip().upper()}: {t['title']}", f"Team: {t['team']}", f"Status: {t['status']}",
                 f"Priority: {t['priority']}", f"Assignee: {t['assignee']}", f"Created: {t['created']}",
                 f"Labels: {', '.join(t['labels']) or '-'}", f"Duplicate of: {t['duplicate_of'] or '-'}",
                 f"SLA due: {t['sla_due']}", f"Customer tier: {t['tier']}",
                 f"Description: {t['description']}", "Comments:"]
        lines += [f"  - {c}" for c in t["comments"]] or ["  (none)"]
        return "\n".join(lines)

    def tool_list_users(self, team: str = "") -> str:
        rows = [f"{n} | {u['team']} | {u['role']} | on-call: {'yes' if u['on_call'] else 'no'} | "
                f"on leave: {'yes' if u['on_leave'] else 'no'}"
                for n, u in sorted(self.users.items()) if not team or u["team"] == team]
        return "name | team | role | on-call | on leave\n" + "\n".join(rows)

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

    def check(self, final: str) -> tuple[bool, str]:
        wrong = []
        for tid, want in self.expected.items():
            have = self.tickets[tid]
            for key in ("status", "assignee", "priority"):
                if have[key] != want[key]:
                    wrong.append(f"{tid}.{key}")
            if sorted(have["labels"]) != sorted(want["labels"]):
                wrong.append(f"{tid}.labels")
            norm = lambda cs: [" ".join(c.strip().rstrip(".").lower().split()) for c in cs]  # noqa: E731
            if norm(have["comments"]) != norm(want["comments"]):
                wrong.append(f"{tid}.comments")
        return not wrong, "tracker state matches" if not wrong else "wrong: " + ", ".join(wrong[:8])

    def oracle(self) -> list[Step]:
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
        return [Step(calls=[Call("list_users", {"team": self.team})]),
                Step(calls=[Call("list_tickets", {"team": self.team})]),
                Step(calls=updates + comments) if updates or comments else Step(calls=[Call("list_tickets", {})]),
                Step(content=f"Done: {len(updates)} updates, {len(comments)} comments.")]
