"""Boundary tests: every list-returning MCP tool must keep its serialized
response under the in-session truncation budget.

Hermes truncates tool results above a context-scaled threshold
(``context_length * 4 chars/token * 0.15`` = 39,321 chars for a 64K-token
model) and the sandbox-persistence fallback is unavailable in this profile, so
an oversized result degrades to a broken 1.5K preview. The tools therefore
cap their own tables (rows pre-sorted by relevance, tail-trimmed) and report
the cap in ``coverage`` so a partial table is never presented as complete.

Boundary values covered: exactly at budget (no cap), just over (cap engages),
huge single row (degrades to empty), multiple list keys, counters intact.
"""

import json
from unittest.mock import AsyncMock

from mcp.client.session import ClientSession

from mcp_tracker.mcp.tools.issue_read import (
    _RESPONSE_BUDGET_CHARS,
    _cap_rows_for_budget,
    _cap_worklogs_for_budget,
)
from mcp_tracker.tracker.proto.types.issues import Issue, StatusReference
from tests.mcp.conftest import get_tool_result_content

BUDGET = _RESPONSE_BUDGET_CHARS


# ── unit: helper boundary behaviour ────────────────────────────────


class TestCapRowsForBudget:
    def test_small_response_untouched(self):
        resp = {"status": "ok", "table": [{"key": "A", "v": "x" * 100}] * 10}
        out = _cap_rows_for_budget(resp, "table")
        assert out is resp
        assert "rows_capped" not in out.get("coverage", {})

    def test_exactly_at_budget_untouched(self):
        resp = {"status": "ok", "table": [{"key": "K", "v": "x" * 80}] * 500}
        # shrink to just under/at budget without ever crossing it
        while len(json.dumps(resp, ensure_ascii=False)) > BUDGET:
            resp["table"] = resp["table"][:-1]
        out = _cap_rows_for_budget(dict(resp), "table")
        assert len(json.dumps(out, ensure_ascii=False)) <= BUDGET
        assert "rows_capped" not in out.get("coverage", {})

    def test_over_budget_capped_with_totals_and_intact_counters(self):
        rows = [
            {"key": f"K{i}", "summary": "s" * 120, "url": f"https://t/{i}"}
            for i in range(300)
        ]
        resp = {
            "status": "complete",
            "issues_total": len(rows),
            "total_hours": 1234.5,
            "table": rows,
            "coverage": {"complete": True},
        }
        out = _cap_rows_for_budget(resp, "table")
        assert len(json.dumps(out, ensure_ascii=False)) <= BUDGET
        assert out["coverage"]["rows_capped"] is True
        assert out["coverage"]["rows_total_table"] == 300
        assert out["coverage"]["rows_returned_table"] == len(out["table"])
        assert out["issues_total"] == 300 and out["total_hours"] == 1234.5
        assert out["table"][0]["key"] == "K0"

    def test_giant_single_row_degrades_to_empty(self):
        rows = [{"key": "K0", "summary": "x" * 60_000}]
        out = _cap_rows_for_budget({"table": rows, "coverage": {}}, "table")
        assert len(json.dumps(out, ensure_ascii=False)) <= BUDGET
        assert out["coverage"]["rows_capped"] is True
        assert out["table"] == []

    def test_multiple_list_keys_all_trimmed(self):
        rows = [{"key": f"K{i}", "v": "z" * 90} for i in range(400)]
        resp = {
            "status": "ok",
            "evidence": rows,
            "all_issue_keys": [f"K{i}" for i in range(400)],
            "coverage": {},
        }
        out = _cap_rows_for_budget(resp, "evidence", "all_issue_keys")
        assert len(json.dumps(out, ensure_ascii=False)) <= BUDGET
        assert out["coverage"]["rows_capped"] is True
        assert out["coverage"]["rows_total_evidence"] == 400
        assert out["coverage"]["rows_total_all_issue_keys"] == 400


class TestCapWorklogsForBudget:
    def test_fits_unchanged(self):
        r = {"T-1": [{"id": 1, "duration": "PT1H"}], "T-2": []}
        out, truncated = _cap_worklogs_for_budget(r)
        assert truncated == {}
        assert out == r

    def test_over_budget_reports_truncated_totals(self):
        entries = [
            {"id": i, "duration": "PT1H", "comment": "c" * 150} for i in range(300)
        ]
        out, truncated = _cap_worklogs_for_budget({"T-1": entries})
        assert len(json.dumps(out, ensure_ascii=False)) <= BUDGET
        assert truncated["T-1"] == 300
        assert len(out["T-1"]) == 0 or len(out["T-1"]) < 300


# ── integration: real tools with boundary-sized mock data ──────────


def _issue(key: str, summary: str = "s" * 60, estimation: str | None = "P1D") -> Issue:
    return Issue.model_construct(
        id=key,
        version=1,
        key=key,
        summary=summary,
        estimation=estimation,
        status=StatusReference.model_construct(id="1", key="open", display="Open"),
    )


def _changelog_with_one_return() -> list[dict]:
    def ev(frm: str, to: str, t: str) -> dict:
        return {
            "updatedAt": t,
            "fields": [
                {
                    "field": {"id": "status", "key": "status"},
                    "from": {"display": frm},
                    "to": {"display": to},
                }
            ],
        }

    return [
        ev("Будем делать", "Тестируется", "2026-01-01T10:00:00"),
        ev("Тестируется", "Провал", "2026-01-01T12:00:00"),
        ev("Провал", "В работе", "2026-01-02T10:00:00"),
        ev("В работе", "Тестируется", "2026-01-03T10:00:00"),
    ]


def _three_pages(issues: list[Issue]) -> list[list[Issue]]:
    return [issues[:100], issues[100:200], issues[200:]]


class TestToolResponsesFitBudget:
    async def test_summarize_effort_caps_huge_release(
        self, client_session: ClientSession, mock_issues_protocol: AsyncMock
    ) -> None:
        issues = [_issue(f"TEST-{i}", summary="s" * 250) for i in range(1, 300)]
        mock_issues_protocol.issues_find_filter = AsyncMock(
            side_effect=_three_pages(issues)
        )

        result = await client_session.call_tool(
            "issues_summarize_effort",
            {
                "scope": "release",
                "measure": "estimation",
                "queue": "TEST",
                "version_id": 1,
            },
        )

        content = get_tool_result_content(result)
        assert len(json.dumps(content, ensure_ascii=False)) <= BUDGET
        assert content["coverage"]["rows_capped"] is True
        assert content["issues_total"] == 299

    async def test_count_release_status_returns_caps(
        self, client_session: ClientSession, mock_issues_protocol: AsyncMock
    ) -> None:
        # exactly at the tool's max_issues=200 boundary: two full pages
        issues = [_issue(f"TEST-{i}", summary="s" * 220) for i in range(1, 201)]
        mock_issues_protocol.issues_find_filter = AsyncMock(
            side_effect=[issues[:100], issues[100:], []]
        )
        mock_issues_protocol.issue_get_status_changelog = AsyncMock(
            return_value=_changelog_with_one_return()
        )

        result = await client_session.call_tool(
            "issues_count_release_status_returns", {"queue": "TEST", "version_id": 1}
        )

        content = get_tool_result_content(result)
        assert len(json.dumps(content, ensure_ascii=False)) <= BUDGET
        assert content["coverage"]["rows_capped"] is True
        assert content["issues_in_release"] == 200
        assert content["total_returns"] == 200

    async def test_comments_wrapped_and_capped(
        self, client_session: ClientSession, mock_issues_protocol: AsyncMock
    ) -> None:
        comments = [{"id": i, "text": "c" * 400} for i in range(300)]
        mock_issues_protocol.issue_get_comments = AsyncMock(return_value=comments)

        result = await client_session.call_tool(
            "issue_get_comments", {"issue_id": "TEST-123"}
        )

        content = get_tool_result_content(result)
        assert isinstance(content, dict)
        assert content["status"] == "complete"
        assert len(json.dumps(content, ensure_ascii=False)) <= BUDGET
        assert content["total_comments"] == 300
        assert content["coverage"]["rows_capped"] is True

    async def test_assigned_open_caps_matches(
        self,
        client_session: ClientSession,
        mock_issues_protocol: AsyncMock,
        mock_users_protocol: AsyncMock,
    ) -> None:
        from mcp_tracker.tracker.proto.types.users import User

        issues = [_issue(f"TEST-{i}", summary="s" * 200) for i in range(1, 300)]
        mock_issues_protocol.issues_find_filter = AsyncMock(
            side_effect=_three_pages(issues)
        )
        mock_users_protocol.users_list = AsyncMock(
            return_value=[User.model_construct(login="tester", display="Tester")]
        )

        result = await client_session.call_tool(
            "issues_assigned_open", {"assignee": "tester"}
        )

        content = get_tool_result_content(result)
        assert len(json.dumps(content, ensure_ascii=False)) <= BUDGET
        assert content["coverage"]["rows_capped"] is True
        assert content["counts"]["open"] >= 299

    async def test_assigned_open_pagination_walks_full_set(
        self,
        client_session: ClientSession,
        mock_issues_protocol: AsyncMock,
        mock_users_protocol: AsyncMock,
    ) -> None:
        from mcp_tracker.tracker.proto.types.users import User

        issues = [_issue(f"TEST-{i}", summary="s" * 200) for i in range(1, 67)]
        mock_users_protocol.users_list = AsyncMock(
            return_value=[User.model_construct(login="tester", display="Tester")]
        )
        mock_issues_protocol.issues_find_filter = AsyncMock(return_value=issues)
        page1 = get_tool_result_content(
            await client_session.call_tool(
                "issues_assigned_open",
                {"assignee": "tester", "page": 1, "per_page": 40},
            )
        )
        mock_issues_protocol.issues_find_filter = AsyncMock(return_value=issues)
        page2 = get_tool_result_content(
            await client_session.call_tool(
                "issues_assigned_open",
                {"assignee": "tester", "page": 2, "per_page": 40},
            )
        )

        assert len(page1["matches"]) == 40
        assert len(page2["matches"]) == 26
        assert page1["coverage"]["total_rows"] == 66
        assert page1["coverage"]["pages_total"] == 2
        assert page1["counts"]["open"] == 66  # global counts on every page
        keys = [m["key"] for m in page1["matches"]] + [
            m["key"] for m in page2["matches"]
        ]
        assert len(set(keys)) == 66  # no overlap, no loss

    async def test_count_release_pagination(
        self, client_session: ClientSession, mock_issues_protocol: AsyncMock
    ) -> None:
        issues = [_issue(f"TEST-{i}", summary="s" * 220) for i in range(1, 201)]
        mock_issues_protocol.issue_get_status_changelog = AsyncMock(
            return_value=_changelog_with_one_return()
        )

        mock_issues_protocol.issues_find_filter = AsyncMock(
            side_effect=[issues[:100], issues[100:], []]
        )
        page1 = get_tool_result_content(
            await client_session.call_tool(
                "issues_count_release_status_returns",
                {
                    "queue": "TEST",
                    "version_id": 1,
                    "include_evidence": False,
                    "per_page": 25,
                },
            )
        )
        mock_issues_protocol.issues_find_filter = AsyncMock(
            side_effect=[issues[:100], issues[100:], []]
        )
        page8 = get_tool_result_content(
            await client_session.call_tool(
                "issues_count_release_status_returns",
                {
                    "queue": "TEST",
                    "version_id": 1,
                    "include_evidence": False,
                    "page": 8,
                    "per_page": 25,
                },
            )
        )

        assert len(page1["table"]) == 25
        assert len(page8["table"]) == 25  # 200 = 8 pages of 25
        assert page1["coverage"]["total_rows"] == 200
        assert page1["coverage"]["pages_total"] == 8
        assert page1["total_returns"] == 200  # counters cover the whole release
        assert page8["total_returns"] == 200
        assert page1["coverage"].get("rows_capped") is None  # page fits, no cap

    async def test_comments_pagination(
        self, client_session: ClientSession, mock_issues_protocol: AsyncMock
    ) -> None:
        comments = [{"id": i, "text": "c" * 200} for i in range(300)]
        mock_issues_protocol.issue_get_comments = AsyncMock(return_value=comments)

        page1 = get_tool_result_content(
            await client_session.call_tool(
                "issue_get_comments",
                {"issue_id": "TEST-123", "page": 1, "per_page": 150},
            )
        )
        page2 = get_tool_result_content(
            await client_session.call_tool(
                "issue_get_comments",
                {"issue_id": "TEST-123", "page": 2, "per_page": 150},
            )
        )

        assert len(page1["comments"]) == 150
        assert len(page2["comments"]) == 150
        assert page1["total_comments"] == 300
        assert page1["coverage"]["pages_total"] == 2
