"""Offline contracts for the bounded, read-only overdue-issues tool."""

from datetime import date

import pytest

from mcp_tracker.mcp.tools.issue_read import issues_overdue_core
from mcp_tracker.tracker.proto.types.issues import Issue
from mcp_tracker.tracker.proto.types.refs import StatusReference, UserReference


def issue(key: str, *, deadline: date | None, status_type: str = "inProgress") -> Issue:
    value = Issue(key=key, summary=f"Summary {key}", deadline=deadline)
    value.status = StatusReference(key="working", display="В работе")
    value.statusType = {"key": status_type, "display": status_type}
    value.assignee = UserReference(uid=1, login="user", display="User")
    return value


class FakeIssues:
    def __init__(self, rows, *, duplicate=False, error=None):
        self.rows = list(rows)
        self.duplicate = duplicate
        self.error = error
        self.calls = []

    async def issues_find_filter(self, filters, *, fields, per_page, page, auth):
        self.calls.append((filters, tuple(fields), per_page, page))
        if self.error:
            raise self.error("boom")
        start = (page - 1) * per_page
        result = self.rows[start : start + per_page]
        if self.duplicate and page == 2:
            return [self.rows[0]]
        return result


@pytest.mark.asyncio
async def test_overdue_semantics_and_complete_pagination():
    rows = [issue(f"YOURQUEUE-{n}", deadline=date(2026, 8, 1)) for n in range(101)]
    rows += [
        issue("YOURQUEUE-200", deadline=date(2026, 8, 25)),
        issue("YOURQUEUE-201", deadline=None),
        issue("YOURQUEUE-202", deadline=date(2026, 7, 1), status_type="done"),
    ]
    fake = FakeIssues(rows)
    result = await issues_overdue_core(fake, object(), queue="yourqueue", as_of="2026-08-25")
    assert result["status"] == "ok"
    assert result["queue"] == "YOURQUEUE"
    assert result["counts"] == {
        "processed": 104,
        "overdue": 101,
        "final_excluded": 1,
        "without_deadline_excluded": 1,
        "unknown_status_type_included_as_open": 0,
    }
    assert result["coverage"] == {
        "complete": True, "reason": None, "pages_fetched": 2, "unique_issues": 104
    }
    assert [call[3] for call in fake.calls] == [1, 2]
    assert all(call[0] == {"queue": "YOURQUEUE"} for call in fake.calls)
    assert result["issues"][0]["url"].endswith("/YOURQUEUE-0")
    assert result["reporting_contract"]["do_not_recalculate_or_group"] is True
    assert result["reporting_contract"]["do_not_add_unsourced_analysis"] is True


@pytest.mark.asyncio
async def test_deadline_equal_to_as_of_is_not_overdue():
    result = await issues_overdue_core(
        FakeIssues([issue("A-1", deadline=date(2026, 8, 25))]),
        object(), queue="A", as_of="2026-08-25",
    )
    assert result["status"] == "no_overdue_issues"
    assert result["counts"]["overdue"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("queue,as_of", [("bad queue", "2026-08-25"), ("YOURQUEUE", "25.08.2026")])
async def test_invalid_input_makes_no_api_call(queue, as_of):
    fake = FakeIssues([])
    result = await issues_overdue_core(fake, object(), queue=queue, as_of=as_of)
    assert result["status"] == "invalid_request"
    assert fake.calls == []


@pytest.mark.asyncio
async def test_upstream_and_duplicate_fail_closed():
    failed = await issues_overdue_core(FakeIssues([], error=TimeoutError), object(), queue="YOURQUEUE")
    assert failed["status"] == "upstream_error"
    assert failed["coverage"]["complete"] is False
    rows = [issue(f"YOURQUEUE-{n}", deadline=date(2026, 1, 1)) for n in range(100)]
    duplicate = await issues_overdue_core(FakeIssues(rows, duplicate=True), object(), queue="YOURQUEUE")
    assert duplicate["status"] == "upstream_error"
    assert "duplicate issue" in duplicate["error"]
