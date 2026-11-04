"""Row-level grading of table fields (shared by the extraction and research families)."""

from __future__ import annotations

import re


def norm(value) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", str(value).lower()).split())


def cell_equal(kind: str, got, want) -> bool:
    if kind == "integer":
        return isinstance(got, int) and not isinstance(got, bool) and got == want
    if kind == "number":
        return isinstance(got, (int, float)) and not isinstance(got, bool) and abs(got - want) <= 0.011
    if kind == "date":
        return str(got).strip() == str(want)
    return norm(got) == norm(want)


def rows_match(have, want: list, key: str, columns: dict) -> list:
    """[rows right, rows expected + rows that should not be there] for a table field. Rows are matched by
    their key column; a row is right when every column matches."""
    if not isinstance(have, list):
        return [0, len(want)]
    got: dict = {}
    for row in have:
        if isinstance(row, dict):
            got.setdefault(norm(row.get(key, "")), []).append(row)
    wanted = {norm(r[key]) for r in want}
    right = sum(any(all(cell_equal(t, r.get(c), row[c]) for c, t in columns.items()) for r in got.get(norm(row[key]), []))
                for row in want)
    extra = sum(len(v) for k, v in got.items() if k not in wanted)
    return [right, len(want) + extra]
